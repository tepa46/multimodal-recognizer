"""
Physics-aware multimodal fusion for fraud detection.

Three variants in one module, selected via the ``variant`` argument
(``rule`` / ``mlp`` / ``gated``). All variants share the same input
schema produced by ``scripts/fusion/generate_multimodal_tuples.py``:

  - roi_features : (B, roi_dim)  visual features pooled per annotation
  - volume_cm3   : (B,)          volume estimated from depth
  - measured_w_g : (B,)          measured weight from scale
  - reported_cls : (B,)          class id predicted by the detector
                                 (or reported by the user, possibly fraudulent)
  - density      : (B,)          density(reported_cls), looked up at runtime

Outputs:
  fraud_logit : (B,)  raw logit; sigmoid → P(fraud)
  loss_dict   : optional, populated during training

The ``rule`` variant has zero learnable parameters and is exposed for
ablation completeness — it implements the threshold-based check used in the
thesis (`|expected - measured| / expected > τ`). ``mlp`` is a vanilla concat
+ MLP baseline. ``gated`` is the main contribution: per-sample modality
gates over (visual, depth, mass) projected into a shared space, plus an
auxiliary physics-aware regulariser L_phys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml


__all__ = [
    "MultimodalFusion",
    "build_fusion_model",
    "load_density_config",
]


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_density_config(path: str | Path) -> dict[str, Any]:
    """Load configs/material_density.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def density_lookup_tensor(
    densities: dict[str, float],
    class_names: list[str] | tuple[str, ...],
    default: float = 0.5,
) -> torch.Tensor:
    """Build a (num_classes,) tensor of densities ordered by ``class_names``."""
    return torch.tensor(
        [float(densities.get(name, default)) for name in class_names],
        dtype=torch.float32,
    )


# ---------------------------------------------------------------------------
# Fusion module
# ---------------------------------------------------------------------------

class MultimodalFusion(nn.Module):
    """Three-variant fusion head for fraud classification.

    Args:
        variant: 'rule' | 'mlp' | 'gated'.
        num_classes: number of object classes (10 for hybrid-v3.3).
        roi_dim: dim of pooled visual features. Default 256 for HTC's
            mask/bbox FPN feature pyramid output.
        density_table: (num_classes,) tensor of densities (g/cm^3) ordered
            by class id. Registered as a buffer.
        threshold: only used by variant='rule'. Searched in eval_fraud.py.
        proj_dim: shared latent size for the gated variant.
    """

    def __init__(
        self,
        variant: str,
        num_classes: int,
        roi_dim: int = 256,
        density_table: Optional[torch.Tensor] = None,
        threshold: float = 0.3,
        proj_dim: int = 64,
        mlp_hidden: int = 128,
        dropout: float = 0.3,
        input_repr: str = "raw",
        use_expected_mass: bool = True,
        use_gate: bool = True,
        gate_input: str = "vdm",
    ) -> None:
        super().__init__()
        if variant not in ("rule", "mlp", "gated"):
            raise ValueError(f"Unknown variant: {variant!r}")
        if input_repr not in ("raw", "log"):
            raise ValueError(f"Unknown input_repr: {input_repr!r}")
        self.variant = variant
        self.input_repr = input_repr
        self.use_expected_mass = bool(use_expected_mass)
        self.use_gate = bool(use_gate)
        if gate_input not in ("vdm", "vdmc", "vdmc+r"):
            raise ValueError(f"Unknown gate_input: {gate_input!r}")
        self.gate_input = gate_input
        self.num_classes = num_classes
        self.roi_dim = roi_dim
        self.threshold = float(threshold)
        self.proj_dim = proj_dim

        if density_table is None:
            density_table = torch.full((num_classes,), 0.5)
        else:
            assert density_table.numel() == num_classes
        self.register_buffer("density_table", density_table.float())

        if variant == "rule":
            return  # no learnable parameters

        n_mass = self._mass_dim()
        scalar_dim = 1 + n_mass
        cls_dim = num_classes

        if variant == "mlp":
            in_dim = roi_dim + scalar_dim + cls_dim
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, mlp_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(mlp_hidden, mlp_hidden // 4),
                nn.ReLU(inplace=True),
                nn.Linear(mlp_hidden // 4, 1),
            )
            return

        # variant == 'gated'
        # Per-modality projections to a shared proj_dim.
        self.visual_proj = nn.Sequential(
            nn.Linear(roi_dim, proj_dim),
            nn.ReLU(inplace=True),
        )
        self.depth_proj = nn.Sequential(
            nn.Linear(1, proj_dim),  # volume only
            nn.ReLU(inplace=True),
        )
        self.mass_proj = nn.Sequential(
            nn.Linear(n_mass, proj_dim),
            nn.ReLU(inplace=True),
        )
        gate_in = proj_dim * (4 if gate_input != "vdm" else 3) + (1 if gate_input == "vdmc+r" else 0)
        self.modality_gate = nn.Sequential(
            nn.Linear(gate_in, 3),
            nn.Sigmoid(),
        )
        # Class branch is added afterwards (used by physics loss too).
        self.class_proj = nn.Sequential(
            nn.Linear(num_classes, proj_dim),
            nn.ReLU(inplace=True),
        )
        self.fraud_head = nn.Sequential(
            nn.Linear(proj_dim * 4, mlp_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )
        self.class_head = nn.Linear(proj_dim * 4, num_classes)

    # ------------------------------------------------------------------
    # Forward variants
    # ------------------------------------------------------------------

    def _mass_dim(self) -> int:
        if self.input_repr == "raw":
            return 2 if self.use_expected_mass else 1
        return 3 if self.use_expected_mass else 1

    def _scalar_inputs(
        self, volume: torch.Tensor, measured: torch.Tensor, expected: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.input_repr == "raw":
            v = volume.unsqueeze(1)
            m = (torch.stack([measured, expected], dim=1) if self.use_expected_mass
                 else measured.unsqueeze(1))
            return v, m
        lv = torch.log(volume.clamp(min=1e-3)).unsqueeze(1)
        lm = torch.log(measured.clamp(min=1e-3))
        if not self.use_expected_mass:
            return lv, lm.unsqueeze(1)
        le = torch.log(expected.clamp(min=1e-3))
        return lv, torch.stack([lm, le, lm - le], dim=1)

    def expected_weight(
        self, volume: torch.Tensor, class_id: torch.Tensor
    ) -> torch.Tensor:
        """expected_w = volume * density(class)."""
        density = self.density_table[class_id]
        return volume * density

    def forward(
        self,
        roi_features: Optional[torch.Tensor],
        volume: torch.Tensor,
        measured_weight: torch.Tensor,
        reported_class: torch.Tensor,
        is_fraud: Optional[torch.Tensor] = None,
        lambda_phys: float = 1.0,
        mu_cls: float = 0.0,
        expected_weight: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            roi_features: (B, roi_dim) — required for mlp/gated, ignored for rule.
            volume: (B,) cm^3.
            measured_weight: (B,) g.
            reported_class: (B,) long, class id reported by the upstream detector.
            is_fraud: (B,) optional binary labels. If supplied, returns a
                ``loss`` key (BCE [+ λ·L_phys (+ μ·CE) for gated]).
            lambda_phys: weight of the physics-aware auxiliary loss.
            mu_cls: weight of the cross-entropy anchor tying the auxiliary
                soft-class distribution q to the reported class on
                legitimate samples (gated only; 0 disables).

        Returns:
            dict with keys: 'fraud_logit' (B,), 'expected_weight' (B,),
            optionally 'loss', 'loss_bce', 'loss_phys', 'loss_cls',
            'modality_gate', 'soft_class', 'soft_expected_weight'.
        """
        if expected_weight is not None:
            expected_w = expected_weight
        else:
            expected_w = self.expected_weight(volume, reported_class)
        out: dict[str, torch.Tensor] = {"expected_weight": expected_w}

        if self.variant == "rule":
            denom = torch.clamp(expected_w, min=1e-3)
            rel_err = (measured_weight - expected_w).abs() / denom
            # Logit for compatibility: stretch rel_err around threshold.
            logit = (rel_err - self.threshold) * 10.0
            out["fraud_logit"] = logit
            out["rel_err"] = rel_err
            if is_fraud is not None:
                out["loss"] = F.binary_cross_entropy_with_logits(
                    logit, is_fraud.float()
                )
                out["loss_bce"] = out["loss"].detach()
            return out

        if roi_features is None:
            raise ValueError("roi_features required for mlp/gated variants")

        cls_onehot = F.one_hot(
            reported_class, num_classes=self.num_classes
        ).float()
        vol_feat, mass_feat = self._scalar_inputs(volume, measured_weight, expected_w)

        if self.variant == "mlp":
            x = torch.cat([roi_features, vol_feat, mass_feat, cls_onehot], dim=1)
            logit = self.mlp(x).squeeze(-1)
            out["fraud_logit"] = logit
            if is_fraud is not None:
                out["loss"] = F.binary_cross_entropy_with_logits(
                    logit, is_fraud.float()
                )
                out["loss_bce"] = out["loss"].detach()
            return out

        # variant == 'gated'
        v = self.visual_proj(roi_features)              # (B, P)
        d = self.depth_proj(vol_feat)
        m = self.mass_proj(mass_feat)
        c = self.class_proj(cls_onehot)                 # (B, P)

        if self.use_gate:
            parts = [v, d, m]
            if self.gate_input != "vdm":
                parts.append(c)
            if self.gate_input == "vdmc+r":
                resid = torch.log(measured_weight.clamp(min=1e-3)) - torch.log(expected_w.clamp(min=1e-3))
                parts.append(resid.unsqueeze(1))
            gate = self.modality_gate(torch.cat(parts, dim=1))
            out["modality_gate"] = gate
            v_g = v * gate[:, 0:1]
            d_g = d * gate[:, 1:2]
            m_g = m * gate[:, 2:3]
        else:
            v_g, d_g, m_g = v, d, m
        fused = torch.cat([v_g, d_g, m_g, c], dim=1)
        logit = self.fraud_head(fused).squeeze(-1)
        out["fraud_logit"] = logit

        # Auxiliary soft-class distribution q from the fused representation.
        # Soft expected mass m_tilde = V * sum_k q_k * rho_k.
        cls_logits = self.class_head(fused)                  # (B, K)
        q = F.softmax(cls_logits, dim=1)                     # (B, K)
        m_tilde = volume * (q @ self.density_table)          # (B,)
        out["soft_class"] = q
        out["soft_expected_weight"] = m_tilde

        if is_fraud is not None:
            bce = F.binary_cross_entropy_with_logits(logit, is_fraud.float())
            out["loss_bce"] = bce.detach()
            mask = (is_fraud < 0.5)
            if mask.any():
                mt = m_tilde[mask]
                meas_w = measured_weight[mask]
                denom = torch.clamp(mt, min=1e-3)
                phys = ((mt - meas_w) / denom).pow(2).mean()
                if mu_cls > 0:
                    ce = F.cross_entropy(
                        cls_logits[mask], reported_class[mask]
                    )
                else:
                    ce = torch.zeros((), device=logit.device)
            else:
                phys = torch.zeros((), device=logit.device)
                ce = torch.zeros((), device=logit.device)
            out["loss_phys"] = phys.detach()
            out["loss_cls"] = ce.detach()
            out["loss"] = bce + lambda_phys * phys + mu_cls * ce

        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

DEFAULT_CLASS_ORDER = (
    "plastic_bottle", "glass_bottle", "glass_jar", "metal_can",
    "carton", "tetrapack", "paper", "plastic_bag",
    "plastic_container", "other",
)


def build_fusion_model(
    variant: str,
    density_yaml: str | Path = "configs/material_density.yaml",
    class_names: tuple[str, ...] = DEFAULT_CLASS_ORDER,
    **kwargs: Any,
) -> MultimodalFusion:
    """Convenience builder used by train/eval scripts."""
    cfg = load_density_config(density_yaml)
    density_table = density_lookup_tensor(cfg["densities"], list(class_names))
    return MultimodalFusion(
        variant=variant,
        num_classes=len(class_names),
        density_table=density_table,
        **kwargs,
    )
