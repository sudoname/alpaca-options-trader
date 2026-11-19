"""Multitask MLP challenger (Oracle 4, Phase 4).

Torch-only building blocks live here: the network, head-label assembly,
the seeded training loop with early stopping, temperature scaling, and
checksum-verified save/load.  The experiment runner
(``oracle.ml.train.run_mlp``) owns data loading, comparators and
reporting, and imports this module lazily so the rest of the research
pipeline works without torch installed.

Research-only: nothing here touches the production trading path.
Honesty rules: missing labels are masked (never imputed as zero), heads
with no valid training labels are excluded from the loss and reported as
unavailable, and a corrupted/mismatched saved model raises
``ModelIntegrityError`` so callers abstain instead of trusting it.

Heads (per master prompt Phase 4):
  * direction   — 3-class logits over {down, flat, up}; the dataset's
                  direction label encodes v in {-1.0, 0.0, +1.0}
                  (labels.py `_direction`), mapped to classes {0, 1, 2}.
  * quantile    — ordered return quantiles over QUANTILE_LADDER via a
                  nonnegative-gap parameterization (q05 + cumsum of
                  softplus gaps); ordered by construction.
  * rv          — log realized-vol regression (SmoothL1 on log scale);
                  forecasts are exp(clamped) hence strictly positive.
  * hazard      — discrete-time competing risks: per time-bin softmax
                  over {survive} + HAZARD_CAUSES, trained on the same
                  person-period expansion as the classical baseline
                  (censored subjects contribute survival periods only).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from oracle.ml.baselines import (HAZARD_BIN_EDGES_MIN, HAZARD_CAUSES,
                                 QUANTILE_LADDER, PersonPeriod,
                                 expand_person_periods)

try:  # pragma: no cover - import guard exercised via monkeypatch in tests
    import torch
    import torch.nn.functional as F
    from torch import nn
    TORCH_AVAILABLE = True
    TORCH_WHY: Optional[str] = None
except Exception as _exc:  # pragma: no cover
    torch = None            # type: ignore[assignment]
    F = None                # type: ignore[assignment]
    nn = None               # type: ignore[assignment]
    TORCH_AVAILABLE = False
    TORCH_WHY = f"torch not importable: {_exc}"

N_HAZARD_BINS = len(HAZARD_BIN_EDGES_MIN) - 1
N_HAZARD_CLASSES = len(HAZARD_CAUSES) + 1          # survive + causes

__all__ = [
    "TORCH_AVAILABLE", "TORCH_WHY", "ModelIntegrityError", "MLPConfig",
    "HeadData", "MultitaskMLP", "assemble_head_data", "direction_class",
    "train_mlp", "fit_temperature", "predict_heads", "cif_from_hazards",
    "hazard_row_nll", "resolve_device", "save_model", "load_model",
    "N_HAZARD_BINS", "N_HAZARD_CLASSES",
]


class ModelIntegrityError(RuntimeError):
    """Saved model artifact is missing, corrupted or schema-mismatched.

    Callers must treat this as 'model unavailable' and abstain; the
    production baseline is unaffected.
    """


@dataclass
class MLPConfig:
    hidden: Tuple[int, int, int] = (128, 64, 32)
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 40
    patience: int = 5
    grad_clip: float = 1.0
    quantiles: Tuple[float, ...] = tuple(QUANTILE_LADDER)


# --------------------------------------------------------------------------- #
# label assembly (framework-free; torch not required)
# --------------------------------------------------------------------------- #

def direction_class(label_value: float) -> int:
    """Map the dataset's direction label {-1, 0, +1} to class {0, 1, 2}."""
    if label_value < 0:
        return 0
    if label_value > 0:
        return 2
    return 1


def _purge_safe(label_end: Optional[datetime],
                label_available_at: Optional[datetime],
                test_start: Optional[datetime]) -> bool:
    """True iff using this label cannot leak information at/after test_start."""
    if test_start is None:
        return True
    if label_end is None or label_end >= test_start:
        return False
    if label_available_at is not None and label_available_at >= test_start:
        return False
    return True


@dataclass
class HeadData:
    """Per-row head targets aligned with the model's input rows.

    ``None`` means masked (missing / censored / purged); masked rows are
    excluded from the loss entirely, never imputed.
    """

    direction_cls: List[Optional[int]] = field(default_factory=list)
    log_return: List[Optional[float]] = field(default_factory=list)
    rv: List[Optional[float]] = field(default_factory=list)
    hazard_time: List[Optional[float]] = field(default_factory=list)
    hazard_cause: List[Optional[str]] = field(default_factory=list)

    def counts(self) -> Dict[str, int]:
        return {
            "direction": sum(1 for v in self.direction_cls if v is not None),
            "quantile": sum(1 for v in self.log_return if v is not None),
            "rv": sum(1 for v in self.rv if v is not None),
            "hazard": sum(1 for v in self.hazard_time if v is not None),
        }


def assemble_head_data(rows: Sequence[Any],
                       aux_logret: Dict[str, Any],
                       aux_rv: Dict[str, Any],
                       aux_event: Dict[str, Any],
                       causes_by_id: Dict[str, str],
                       test_start: Optional[datetime],
                       include_censored_hazard: bool = False) -> HeadData:
    """Build per-head targets for ``rows`` (direction Samples).

    * direction comes from the row's own label (status 'ok' only).
    * log_return / future_rv / first_touch come from aux Samples joined
      by decision_id (missing join => masked).
    * every label is additionally gated by ``_purge_safe`` against
      ``test_start`` so no training/validation label overlaps the
      evaluation window.
    * hazard: events need label_value > 0 minutes and a known cause in
      HAZARD_CAUSES; censored subjects (only when
      ``include_censored_hazard``) contribute survival time
      (label_end - as_of) and cause ``None`` — the person-period
      expansion turns that into survival-only periods, matching the
      Phase 3 baseline exactly.
    """
    hd = HeadData()
    for s in rows:
        # direction (the row's own label)
        d_cls: Optional[int] = None
        if (s.label_status == "ok" and s.label_value is not None
                and _purge_safe(s.label_end, s.label_available_at,
                                test_start)):
            d_cls = direction_class(float(s.label_value))
        hd.direction_cls.append(d_cls)

        # log return (quantile head)
        lr_v: Optional[float] = None
        a = aux_logret.get(s.decision_id)
        if (a is not None and a.label_status == "ok"
                and a.label_value is not None
                and _purge_safe(a.label_end, a.label_available_at,
                                test_start)):
            lr_v = float(a.label_value)
        hd.log_return.append(lr_v)

        # future realized vol (rv head); must be strictly positive for
        # the log-scale loss
        rv_v: Optional[float] = None
        a = aux_rv.get(s.decision_id)
        if (a is not None and a.label_status == "ok"
                and a.label_value is not None and float(a.label_value) > 0
                and _purge_safe(a.label_end, a.label_available_at,
                                test_start)):
            rv_v = float(a.label_value)
        hd.rv.append(rv_v)

        # hazard (first-touch competing risks)
        t_v: Optional[float] = None
        c_v: Optional[str] = None
        ev = aux_event.get(s.decision_id)
        if ev is not None:
            cause = causes_by_id.get(s.decision_id)
            if (ev.label_status == "ok" and ev.label_value is not None
                    and float(ev.label_value) > 0
                    and cause in HAZARD_CAUSES
                    and _purge_safe(ev.label_end, ev.label_available_at,
                                    test_start)):
                t_v, c_v = float(ev.label_value), cause
            elif (include_censored_hazard and ev.label_status == "censored"
                    and ev.label_end is not None and ev.as_of is not None
                    and _purge_safe(ev.label_end, ev.label_available_at,
                                    test_start)):
                mins = (ev.label_end - ev.as_of).total_seconds() / 60.0
                if mins > 0:
                    t_v, c_v = mins, None
        hd.hazard_time.append(t_v)
        hd.hazard_cause.append(c_v)
    return hd


# --------------------------------------------------------------------------- #
# naive-loss normalizers (train split only; used to balance head losses)
# --------------------------------------------------------------------------- #

def _smooth_l1_np(diff: np.ndarray, beta: float = 1.0) -> np.ndarray:
    a = np.abs(diff)
    return np.where(a < beta, 0.5 * a * a / beta, a - 0.5 * beta)


def _marginal_hazard_table(periods: Sequence[PersonPeriod],
                           eps: float = 1e-6) -> np.ndarray:
    counts = np.zeros((N_HAZARD_BINS, N_HAZARD_CLASSES))
    for pp in periods:
        counts[pp.time_bin, pp.outcome] += 1.0
    denom = counts.sum(axis=1, keepdims=True) + eps * N_HAZARD_CLASSES
    return (counts + eps) / denom


def _naive_normalizers(head: HeadData,
                       taus: Sequence[float]) -> Dict[str, Optional[float]]:
    """Per-head naive loss scale from TRAIN labels only.

    direction: entropy of the train class distribution.
    quantile: marginal (unconditional) train pinball.
    rv: SmoothL1 of log y around its train mean.
    hazard: marginal per-bin table NLL on train person-periods.
    A head with no train labels gets ``None`` => excluded from the loss.
    """
    out: Dict[str, Optional[float]] = {}

    cls = [c for c in head.direction_cls if c is not None]
    if cls:
        counts = np.bincount(np.asarray(cls), minlength=3).astype(float)
        p = counts / counts.sum()
        ent = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
        out["direction"] = max(ent, 1e-6)
    else:
        out["direction"] = None

    y = np.asarray([v for v in head.log_return if v is not None])
    if y.size:
        t = np.asarray(taus)
        qs = np.quantile(y, t)
        d = y[:, None] - qs[None, :]
        pb = float(np.maximum(t[None, :] * d, (t[None, :] - 1.0) * d).mean())
        out["quantile"] = max(pb, 1e-6)
    else:
        out["quantile"] = None

    rv = np.asarray([v for v in head.rv if v is not None])
    if rv.size:
        logy = np.log(rv)
        out["rv"] = max(float(_smooth_l1_np(logy - logy.mean()).mean()), 1e-6)
    else:
        out["rv"] = None

    periods = expand_person_periods(head.hazard_time, head.hazard_cause)
    if periods:
        table = _marginal_hazard_table(periods)
        nll = float(np.mean([-math.log(table[pp.time_bin, pp.outcome])
                             for pp in periods]))
        out["hazard"] = max(nll, 1e-6)
    else:
        out["hazard"] = None
    return out


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

if TORCH_AVAILABLE:

    class MultitaskMLP(nn.Module):
        """Compact shared-trunk multitask MLP (Phase 4 spec)."""

        def __init__(self, d_in: int, cfg: Optional[MLPConfig] = None
                     ) -> None:
            super().__init__()
            cfg = cfg or MLPConfig()
            h1, h2, h3 = cfg.hidden
            self.d_in = d_in
            self.cfg = cfg
            self.trunk = nn.Sequential(
                nn.Linear(d_in, h1), nn.LayerNorm(h1), nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(h1, h2), nn.GELU(), nn.Dropout(cfg.dropout),
                nn.Linear(h2, h3),
            )
            self.head_direction = nn.Linear(h3, 3)
            self.head_quantile = nn.Linear(h3, len(cfg.quantiles))
            self.head_rv = nn.Linear(h3, 1)
            self.head_hazard = nn.Linear(h3,
                                         N_HAZARD_BINS * N_HAZARD_CLASSES)

        def forward(self, x: "torch.Tensor") -> Dict[str, "torch.Tensor"]:
            z = self.trunk(x)
            raw_q = self.head_quantile(z)
            # ordered by construction: q05 + cumulative softplus gaps
            q = torch.cat(
                [raw_q[:, :1],
                 raw_q[:, :1] + torch.cumsum(F.softplus(raw_q[:, 1:]),
                                             dim=1)],
                dim=1)
            return {
                "direction_logits": self.head_direction(z),
                "quantiles": q,
                "rv_log": self.head_rv(z).squeeze(-1),
                "hazard_logits": self.head_hazard(z).view(
                    -1, N_HAZARD_BINS, N_HAZARD_CLASSES),
            }

else:  # pragma: no cover - placeholder keeps attribute access explicit
    MultitaskMLP = None  # type: ignore[assignment]


def resolve_device(device: str) -> str:
    """'auto' => cuda when available else cpu; otherwise pass through."""
    if device == "auto":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return device


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #

def _head_tensors(head: HeadData, dev: "torch.device") -> Dict[str, Any]:
    dir_idx = [i for i, c in enumerate(head.direction_cls) if c is not None]
    dir_cls = [head.direction_cls[i] for i in dir_idx]
    q_idx = [i for i, v in enumerate(head.log_return) if v is not None]
    q_y = [head.log_return[i] for i in q_idx]
    rv_idx = [i for i, v in enumerate(head.rv) if v is not None]
    rv_logy = [math.log(head.rv[i]) for i in rv_idx]
    periods = expand_person_periods(head.hazard_time, head.hazard_cause)
    return {
        "dir_idx": torch.tensor(dir_idx, dtype=torch.long, device=dev),
        "dir_cls": torch.tensor(dir_cls, dtype=torch.long, device=dev),
        "q_idx": torch.tensor(q_idx, dtype=torch.long, device=dev),
        "q_y": torch.tensor(q_y, dtype=torch.float32, device=dev),
        "rv_idx": torch.tensor(rv_idx, dtype=torch.long, device=dev),
        "rv_logy": torch.tensor(rv_logy, dtype=torch.float32, device=dev),
        "hz_row": torch.tensor([pp.row_index for pp in periods],
                               dtype=torch.long, device=dev),
        "hz_bin": torch.tensor([pp.time_bin for pp in periods],
                               dtype=torch.long, device=dev),
        "hz_out": torch.tensor([pp.outcome for pp in periods],
                               dtype=torch.long, device=dev),
    }


def _class_weights(head: HeadData,
                   dev: "torch.device") -> Optional["torch.Tensor"]:
    """Inverse-frequency class weights from TRAIN direction labels only."""
    cls = [c for c in head.direction_cls if c is not None]
    if not cls:
        return None
    counts = np.bincount(np.asarray(cls), minlength=3).astype(float)
    present = counts > 0
    w = np.zeros(3)
    w[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(w, dtype=torch.float32, device=dev)


def _composite_loss(out: Dict[str, "torch.Tensor"],
                    rows: "torch.Tensor",
                    ht: Dict[str, Any],
                    class_w: Optional["torch.Tensor"],
                    taus: "torch.Tensor",
                    norms: Dict[str, Optional[float]],
                    n_total: int
                    ) -> Tuple[Optional["torch.Tensor"],
                               Dict[str, Optional[float]], Dict[str, int]]:
    """Weighted multitask loss over the rows in this (mini)batch.

    Each available head loss is divided by its train-derived naive loss
    so heads contribute on comparable scales; heads with no labelled
    items in the batch (or no train labels at all) contribute nothing.
    """
    dev = rows.device
    pos = torch.full((n_total,), -1, dtype=torch.long, device=dev)
    pos[rows] = torch.arange(rows.shape[0], device=dev)

    total: Optional["torch.Tensor"] = None
    heads: Dict[str, Optional[float]] = {}
    items: Dict[str, int] = {}

    def add(name: str, loss: Optional["torch.Tensor"], count: int) -> None:
        nonlocal total
        heads[name] = float(loss.detach()) if loss is not None else None
        items[name] = count
        norm = norms.get(name)
        if loss is None or norm is None:
            return
        scaled = loss / norm
        total = scaled if total is None else total + scaled

    # direction: weighted cross-entropy over 3 classes
    loss, count = None, 0
    if ht["dir_idx"].numel():
        sel = pos[ht["dir_idx"]]
        m = sel >= 0
        count = int(m.sum())
        if count:
            loss = F.cross_entropy(out["direction_logits"][sel[m]],
                                   ht["dir_cls"][m], weight=class_w)
    add("direction", loss, count)

    # quantile: mean pinball across the ladder
    loss, count = None, 0
    if ht["q_idx"].numel():
        sel = pos[ht["q_idx"]]
        m = sel >= 0
        count = int(m.sum())
        if count:
            pred = out["quantiles"][sel[m]]
            d = ht["q_y"][m].unsqueeze(1) - pred
            loss = torch.maximum(taus * d, (taus - 1.0) * d).mean()
    add("quantile", loss, count)

    # rv: SmoothL1 on the log scale (positive forecast by construction)
    loss, count = None, 0
    if ht["rv_idx"].numel():
        sel = pos[ht["rv_idx"]]
        m = sel >= 0
        count = int(m.sum())
        if count:
            loss = F.smooth_l1_loss(out["rv_log"][sel[m]], ht["rv_logy"][m])
    add("rv", loss, count)

    # hazard: person-period NLL (cross-entropy over survive+causes)
    loss, count = None, 0
    if ht["hz_row"].numel():
        sel = pos[ht["hz_row"]]
        m = sel >= 0
        count = int(m.sum())
        if count:
            logits = out["hazard_logits"][sel[m], ht["hz_bin"][m]]
            loss = F.cross_entropy(logits, ht["hz_out"][m])
    add("hazard", loss, count)

    return total, heads, items


def train_mlp(x_train: np.ndarray, head_train: HeadData,
              x_val: np.ndarray, head_val: HeadData,
              cfg: Optional[MLPConfig] = None, seed: int = 42,
              device: str = "cpu"
              ) -> Tuple["MultitaskMLP", Dict[str, Any]]:
    """Train the multitask MLP with early stopping on the validation
    composite loss (same train-derived normalizers for train and val).

    Returns (model_with_best_weights, history dict).
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError(TORCH_WHY or "torch unavailable")
    if x_train.ndim != 2 or x_train.shape[0] == 0:
        raise ValueError("x_train must be a non-empty (n, d) matrix")
    cfg = cfg or MLPConfig()
    dev = torch.device(device)
    torch.manual_seed(seed)

    n, d_in = x_train.shape
    model = MultitaskMLP(d_in, cfg).to(dev)
    xt = torch.as_tensor(np.asarray(x_train, dtype=np.float32), device=dev)
    xv = torch.as_tensor(np.asarray(x_val, dtype=np.float32), device=dev)
    ht_tr = _head_tensors(head_train, dev)
    ht_va = _head_tensors(head_val, dev)
    norms = _naive_normalizers(head_train, cfg.quantiles)
    class_w = _class_weights(head_train, dev)
    taus = torch.tensor(cfg.quantiles, dtype=torch.float32,
                        device=dev).unsqueeze(0)

    if all(v is None for v in norms.values()):
        raise ValueError("no head has any usable training label")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    gen = torch.Generator().manual_seed(seed)       # CPU shuffle generator
    n_val = int(xv.shape[0])
    all_val = torch.arange(n_val, device=dev)

    best_val = float("inf")
    best_state: Optional[Dict[str, Any]] = None
    best_epoch = -1
    best_train = None
    best_val_heads: Dict[str, Optional[float]] = {}
    bad = 0
    epochs_run = 0

    for epoch in range(cfg.max_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        ep_loss, ep_batches = 0.0, 0
        for start in range(0, n, cfg.batch_size):
            rows = perm[start:start + cfg.batch_size].to(dev)
            out = model(xt[rows])
            total, _, _ = _composite_loss(out, rows, ht_tr, class_w, taus,
                                          norms, n)
            if total is None:
                continue
            opt.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            ep_loss += float(total.detach())
            ep_batches += 1
        epochs_run = epoch + 1
        train_loss = ep_loss / max(ep_batches, 1)

        model.eval()
        with torch.no_grad():
            if n_val:
                vout = model(xv)
                vtotal, vheads, _ = _composite_loss(vout, all_val, ht_va,
                                                    class_w, taus, norms,
                                                    n_val)
                val_loss = (float(vtotal) if vtotal is not None
                            else float("inf"))
            else:                          # no val rows: fall back to train
                vheads = {}
                val_loss = train_loss

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_epoch = epoch
            best_train = train_loss
            best_val_heads = vheads
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    stopped_early = epochs_run < cfg.max_epochs
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    history = {
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "train_loss_at_best": best_train,
        "val_loss_at_best": (best_val if best_val != float("inf") else None),
        "train_val_gap": ((best_val - best_train)
                          if (best_train is not None
                              and best_val != float("inf")) else None),
        "val_head_losses_at_best": best_val_heads,
        "head_label_counts_train": head_train.counts(),
        "head_label_counts_val": head_val.counts(),
        "normalizers": norms,
        "class_weights": (class_w.tolist() if class_w is not None else None),
        "device": str(dev),
        "seed": seed,
    }
    return model, history


# --------------------------------------------------------------------------- #
# calibration & inference
# --------------------------------------------------------------------------- #

def fit_temperature(logits: np.ndarray, targets: np.ndarray,
                    iters: int = 200, lr: float = 0.1) -> float:
    """Scalar temperature minimizing CE on held-out validation labels.

    Returns 1.0 when fewer than two classes are present (nothing to fit).
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError(TORCH_WHY or "torch unavailable")
    targets = np.asarray(targets, dtype=np.int64)
    if targets.size == 0 or len(set(targets.tolist())) < 2:
        return 1.0
    lg = torch.as_tensor(np.asarray(logits, dtype=np.float32))
    tg = torch.as_tensor(targets)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([log_t], lr=lr)
    for _ in range(iters):
        opt.zero_grad()
        loss = F.cross_entropy(lg / log_t.exp(), tg)
        loss.backward()
        opt.step()
        with torch.no_grad():
            log_t.clamp_(math.log(0.05), math.log(20.0))
    return float(log_t.exp())


def predict_heads(model: "MultitaskMLP", x: np.ndarray, device: str = "cpu",
                  temperature: float = 1.0, batch_size: int = 4096
                  ) -> Dict[str, np.ndarray]:
    """Deterministic batched inference; returns numpy arrays.

    direction_probs (n,3) softmax(logits/T); p_up = P(class up);
    quantiles (n, len(ladder)) ordered; rv_forecast (n,) > 0;
    hazards (n, bins, classes) per-bin softmax; direction_logits raw.
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError(TORCH_WHY or "torch unavailable")
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()
    t = max(float(temperature), 1e-3)
    chunks: Dict[str, List[np.ndarray]] = {
        "direction_logits": [], "direction_probs": [], "quantiles": [],
        "rv_forecast": [], "hazards": []}
    xf = np.asarray(x, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(xf), batch_size):
            xb = torch.as_tensor(xf[start:start + batch_size], device=dev)
            out = model(xb)
            probs = F.softmax(out["direction_logits"] / t, dim=1)
            hz = F.softmax(out["hazard_logits"], dim=2)
            rv = torch.exp(out["rv_log"].clamp(-20.0, 10.0))
            chunks["direction_logits"].append(
                out["direction_logits"].cpu().numpy())
            chunks["direction_probs"].append(probs.cpu().numpy())
            chunks["quantiles"].append(out["quantiles"].cpu().numpy())
            chunks["rv_forecast"].append(rv.cpu().numpy())
            chunks["hazards"].append(hz.cpu().numpy())
    res = {k: (np.concatenate(v) if v else np.zeros((0,)))
           for k, v in chunks.items()}
    res["p_up"] = (res["direction_probs"][:, 2]
                   if len(res["direction_probs"]) else np.zeros((0,)))
    return res


def cif_from_hazards(hazards: np.ndarray) -> np.ndarray:
    """Cumulative incidence per cause from per-bin hazards.

    hazards: (n, n_bins, n_classes) with class 0 = survive.
    Returns (n, n_bins, n_causes); nondecreasing in the bin axis and
    sum over causes <= 1 by construction (same recursion as the
    classical baseline).
    """
    hz = np.asarray(hazards, dtype=float)
    n = hz.shape[0]
    k_causes = hz.shape[2] - 1
    surv = np.ones(n)
    acc = np.zeros((n, k_causes))
    cif = np.zeros((n, hz.shape[1], k_causes))
    for k in range(hz.shape[1]):
        acc = acc + surv[:, None] * hz[:, k, 1:]
        cif[:, k, :] = acc
        surv = surv * hz[:, k, 0]
    return cif


def hazard_row_nll(hazards: np.ndarray, periods: Sequence[PersonPeriod],
                   eps: float = 1e-12) -> Dict[int, Tuple[float, int]]:
    """Per-subject (sum NLL, n_periods) under model hazards."""
    out: Dict[int, Tuple[float, int]] = {}
    for pp in periods:
        p = float(hazards[pp.row_index, pp.time_bin, pp.outcome])
        s, c = out.get(pp.row_index, (0.0, 0))
        out[pp.row_index] = (s - math.log(p + eps), c + 1)
    return out


# --------------------------------------------------------------------------- #
# save / load with integrity checking
# --------------------------------------------------------------------------- #

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def save_model(model: "MultitaskMLP", out_stem: str, *,
               feature_columns: Sequence[str],
               standardizer_state: Dict[str, Any],
               temperature: float, seed: int,
               git_sha: Optional[str], dataset_hash: Optional[str],
               train_sessions: Sequence[str], val_session: Optional[str],
               extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Persist weights + manifest; returns the manifest dict.

    The .pt payload contains only tensors/primitives so it loads with
    ``torch.load(..., weights_only=True)``.  The manifest records a
    sha256 checksum of the saved file plus the full feature schema and
    scaler state needed to reproduce inference exactly.
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError(TORCH_WHY or "torch unavailable")
    cfg = model.cfg
    model_path = f"{out_stem}_model.pt"
    manifest_path = f"{out_stem}_manifest.json"
    payload = {
        "state_dict": model.state_dict(),
        "d_in": int(model.d_in),
        "hidden": [int(h) for h in cfg.hidden],
        "dropout": float(cfg.dropout),
        "quantiles": [float(q) for q in cfg.quantiles],
    }
    torch.save(payload, model_path)
    manifest = {
        "schema_version": 1,
        "model_file": os.path.basename(model_path),
        "checksum_sha256": _sha256_file(model_path),
        "d_in": int(model.d_in),
        "config": asdict(cfg),
        "feature_columns": list(feature_columns),
        "standardizer": standardizer_state,
        "temperature": float(temperature),
        "seed": int(seed),
        "git_sha": git_sha,
        "dataset_hash": dataset_hash,
        "train_sessions": list(train_sessions),
        "val_session": val_session,
        "torch_version": torch.__version__,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        manifest["extra"] = extra
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def standardizer_state(std: Any) -> Dict[str, Any]:
    """Serializable snapshot of a purged_validation.Standardizer."""
    return {
        "feature_names": list(std.feature_names),
        "mean": {k: float(v) for k, v in std.mean.items()},
        "std": {k: float(v) for k, v in std.std.items()},
    }


def load_model(manifest_path: str, device: str = "cpu"
               ) -> Tuple["MultitaskMLP", Dict[str, Any]]:
    """Load a saved model after verifying checksum + schema.

    Raises ModelIntegrityError on any mismatch so callers abstain.
    Only trusted local artifacts are loaded (weights_only=True).
    """
    if not TORCH_AVAILABLE:  # pragma: no cover
        raise RuntimeError(TORCH_WHY or "torch unavailable")
    try:
        with open(manifest_path, encoding="utf-8") as f:
            manifest = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelIntegrityError(f"manifest unreadable: {exc}") from exc

    model_path = os.path.join(os.path.dirname(os.path.abspath(manifest_path)),
                              str(manifest.get("model_file", "")))
    if not os.path.isfile(model_path):
        raise ModelIntegrityError(f"model file missing: {model_path}")
    actual = _sha256_file(model_path)
    if actual != manifest.get("checksum_sha256"):
        raise ModelIntegrityError(
            "checksum mismatch: expected "
            f"{manifest.get('checksum_sha256')}, got {actual}")
    try:
        payload = torch.load(model_path, map_location="cpu",
                             weights_only=True)
    except Exception as exc:
        raise ModelIntegrityError(f"model file unloadable: {exc}") from exc

    d_in = int(payload.get("d_in", -1))
    if d_in != int(manifest.get("d_in", -2)):
        raise ModelIntegrityError("d_in mismatch between payload and "
                                  "manifest")
    if d_in != len(manifest.get("feature_columns", [])):
        raise ModelIntegrityError("feature schema length does not match "
                                  "model input dimension")
    cfg = MLPConfig(hidden=tuple(int(h) for h in payload["hidden"]),
                    dropout=float(payload["dropout"]),
                    quantiles=tuple(float(q) for q in payload["quantiles"]))
    model = MultitaskMLP(d_in, cfg)
    try:
        model.load_state_dict(payload["state_dict"])
    except Exception as exc:
        raise ModelIntegrityError(f"state_dict mismatch: {exc}") from exc
    model.to(torch.device(device))
    model.eval()
    return model, manifest
