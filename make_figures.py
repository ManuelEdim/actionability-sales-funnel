"""Regenerates manuscript figures that are not produced by run_pipeline.py.

Usage:
    python make_figures.py --smartconvert training_sample.csv \
        --balabit Mouse-Dynamics-Challenge/training_files --out results
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import run_pipeline as R


def fig_feature_lift(out):
    lift = pd.read_csv(out / "feature_lift_audit.csv")
    lift = lift[~lift.constant_after_scoping].sort_values("lift")
    colours = ["#c0392b" if f == R.LEAK else ("#9e9e9e" if not r else "#3b6ea8")
               for f, r in zip(lift.feature, lift.in_register)]
    fig, ax = plt.subplots(figsize=(5.2, 5.6))
    ax.barh(lift.feature, lift.lift, color=colours)
    ax.axvline(1, color="black", lw=0.6, ls="--")
    ax.set_xlabel("Lift (prevalence | ordered=1 / prevalence | ordered=0)")
    ax.tick_params(axis="y", labelsize=7)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#c0392b", label="excluded: near-terminal"),
                       Patch(color="#3b6ea8", label="retained"),
                       Patch(color="#9e9e9e", label="screened and excluded")],
              fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(out / "fig_feature_lift.png", dpi=200); plt.close(fig)


def fig_coefficients(scoped, out):
    X = scoped[R.FINAL].to_numpy(float); y = scoped[R.TARGET].to_numpy()
    Xtr, _, ytr, _ = train_test_split(X, y, test_size=R.CFG["validation_fraction"],
                                      random_state=R.SEED, stratify=y)
    val = R.make_model(R.CFG["class_weight"]).fit(Xtr, ytr).coef_[0]
    full = R.make_model(R.CFG["class_weight"]).fit(X, y).coef_[0]
    order = np.argsort(val)
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    ax.barh(np.array(R.FINAL)[order], val[order],
            color=["#c0392b" if v < 0 else "#3b6ea8" for v in val[order]])
    ax.axvline(0, color="black", lw=0.6)
    ax.set_xlabel("Logistic regression coefficient (validation fit)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_coefficients.png", dpi=200); plt.close(fig)
    return {"max_abs_refit_difference": float(np.max(np.abs(val - full))),
            "sign_agreement": bool(np.all(np.sign(val) == np.sign(full))),
            "validation_coefficients": dict(zip(R.FINAL, map(float, val)))}


def fig_balabit_users(balabit_dir, out):
    vmax = R.CFG["balabit"]["max_velocity_px_s"]
    rows = []
    for f in sorted(Path(balabit_dir).glob("user*/session_*")):
        d = pd.read_csv(f); d = d[d["state"].isin(["Move", "Drag"])]
        t, x, y = (d[c].to_numpy(float) for c in ("client timestamp", "x", "y"))
        dt, dx, dy = np.diff(t), np.diff(x), np.diff(y)
        dist = np.hypot(dx, dy); ok = (dt > 0) & (dist > 0)
        v = dist[ok] / dt[ok]; keep = v <= vmax
        h = np.arctan2(dy[ok][keep], dx[ok][keep])
        turn = np.abs(np.angle(np.exp(1j * np.diff(h))))
        rows.append({"user": f.parent.name, "v": v[keep], "j": turn})
    users = sorted({r["user"] for r in rows}, key=lambda u: int(u[4:]))
    vm = [np.concatenate([r["v"] for r in rows if r["user"] == u]).mean() for u in users]
    jm = [np.concatenate([r["j"] for r in rows if r["user"] == u]).mean() for u in users]
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    axes[0].barh(users, vm, color="#3b6ea8"); axes[0].set_xlabel("Mean velocity (px/s)")
    axes[1].barh(users, jm, color="#d9822b"); axes[1].set_xlabel("Mean angular change (rad)")
    for a in axes: a.tick_params(labelsize=7)
    fig.tight_layout(); fig.savefig(out / "fig_balabit_users.png", dpi=200); plt.close(fig)


def fig_pipeline(out):
    fig, ax = plt.subplots(figsize=(12, 4.6)); ax.set_xlim(0, 12); ax.set_ylim(0, 4.6); ax.axis("off")
    col = {"obs": "#d8efd3", "proc": "#d6e4f5", "assume": "#fdebd0", "est": "#e8daef"}

    def box(x, y, w, h, text, kind, dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03",
                                    fc=col[kind], ec="#555", lw=0.8,
                                    ls="--" if dashed else "-"))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7.2, wrap=True)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=0.8, color="#333"))

    box(0.1, 3.2, 1.9, 1.1, "SmartConvert\n455,401 labelled sessions\n23 binary inputs", "obs")
    box(0.1, 0.4, 1.9, 1.1, "Balabit training files\n10 users, 65 sessions\nmove/drag events", "obs")
    box(2.4, 3.4, 2.0, 0.8, "Scope: saw_checkout = 1\n36,471 sessions (52.35%)", "proc")
    box(2.4, 2.3, 2.0, 0.8, "Availability register\n(all 23 inputs; Eq. 2)", "proc")
    box(2.4, 0.5, 2.0, 0.9, "Velocity, angular change\nfilter: dt>0, v<=10,000 px/s", "proc")
    box(4.8, 3.4, 2.2, 0.8, "Reward prior (unweighted LR)\nablations, leave-one-out", "proc")
    box(4.8, 2.3, 2.2, 0.8, "Calibration: Brier, ECE,\nslope, reliability (OOF)", "proc")
    box(4.8, 1.2, 2.2, 0.8, "Five hesitation flags -> tier c\nband [c/6,(c+1)/6]", "assume", True)
    box(4.8, 0.1, 2.2, 0.8, "Shared percentile u -> J, V\nF = 0.6J + 0.4V, top 30%", "assume", True)
    box(7.4, 2.7, 2.1, 1.0, "Targeting diagnostic\nfriction vs random vs score", "est")
    box(7.4, 1.2, 2.1, 1.0, "Sensitivity:\nkappa x epsilon, bootstrap", "est")
    box(9.9, 1.8, 2.0, 1.4, "Conditional model-score\nshifts for declared\nepsilon in [0, 0.30]\n(not causal effects)", "est")
    arrow(2.0, 3.75, 2.4, 3.8); arrow(3.4, 3.4, 3.4, 3.1); arrow(4.4, 2.7, 4.8, 3.6)
    arrow(4.4, 2.7, 4.8, 2.7); arrow(2.0, 0.95, 2.4, 0.95); arrow(4.4, 0.95, 4.8, 0.5)
    arrow(5.9, 1.2, 5.9, 0.9); arrow(7.0, 3.6, 7.4, 3.3); arrow(7.0, 0.5, 7.4, 1.5)
    arrow(7.0, 0.5, 7.4, 3.0); arrow(7.0, 3.6, 7.4, 1.8); arrow(9.5, 1.7, 9.9, 2.3)
    arrow(9.5, 3.3, 9.9, 2.8); arrow(4.4, 2.4, 4.8, 1.6)
    for i, (k, lab) in enumerate([("obs", "observed public data"), ("proc", "reproducible processing"),
                                  ("assume", "declared assumption"), ("est", "assumption-conditional output")]):
        ax.add_patch(FancyBboxPatch((5.0 + i * 1.75, 4.3), 0.18, 0.18, boxstyle="square,pad=0",
                                    fc=col[k], ec="#555", lw=0.6))
        ax.text(5.22 + i * 1.75, 4.39, lab, fontsize=6, va="center")
    fig.tight_layout(); fig.savefig(out / "fig_pipeline.png", dpi=220); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smartconvert", required=True)
    ap.add_argument("--balabit", required=True)
    ap.add_argument("--out", default="results")
    a = ap.parse_args(); out = Path(a.out)
    df = pd.read_csv(a.smartconvert); scoped = df[df[R.SCOPE].eq(1)]
    fig_feature_lift(out)
    coef = fig_coefficients(scoped, out)
    fig_balabit_users(a.balabit, out)
    fig_pipeline(out)
    (out / "coefficient_stability.json").write_text(json.dumps(coef, indent=2))
    print(json.dumps({k: coef[k] for k in ("max_abs_refit_difference", "sign_agreement")}))


if __name__ == "__main__":
    main()
