#!/usr/bin/env python3
"""
ARCAM-Δχ — render 3D da malha micelial e animação dos picos.

Propaga Fe, P e Zn no grafo 3D e mostra o disparo das portas Ln
(aberto = Fe∧P∧¬Zn, bloqueado = Zn).

Dependências: numpy, matplotlib, networkx
  (opcional: ffmpeg para mp4)

Uso:
  python animar_rede_hifal.py
  python animar_rede_hifal.py --nodes 48 --steps 1600 --stride 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simular_rede_hifal import HyphalTissue, Params  # noqa: E402


def record(tissue: HyphalTissue, stride: int) -> dict:
    p = tissue.p
    n_frames = (p.steps + stride - 1) // stride
    n = p.n_nodes
    fe = np.zeros((n_frames, n))
    ph = np.zeros((n_frames, n))
    zn = np.zeros((n_frames, n))
    gate = np.zeros((n_frames, n))
    fired = np.zeros((n_frames, n), dtype=bool)
    times = np.zeros(n_frames)
    last_count = 0
    f = 0
    for k in range(p.steps):
        t = k * p.dt
        tissue.step(t, k)
        if k % stride == 0 and f < n_frames:
            fe[f] = tissue.fe
            ph[f] = tissue.ph
            zn[f] = tissue.zn
            gate[f] = tissue.I_gate
            times[f] = t
            now = len(tissue.spikes)
            if now > last_count:
                for _, i in tissue.spikes[last_count:]:
                    fired[f, i] = True
                last_count = now
            f += 1
    return {
        "fe": fe[:f],
        "ph": ph[:f],
        "zn": zn[:f],
        "gate": gate[:f],
        "fired": fired[:f],
        "times": times[:f],
    }


def node_colors(fe: np.ndarray, ph: np.ndarray, zn: np.ndarray) -> np.ndarray:
    r = np.clip(fe / 1.8, 0.0, 1.0)
    g = np.clip(0.15 + 0.75 * zn / 1.8, 0.0, 1.0)
    b = np.clip(ph / 1.8, 0.0, 1.0)
    rgb = np.stack([r, g * 0.55 + 0.12, b], axis=1)
    lum = rgb.max(axis=1, keepdims=True)
    rgb = np.where(lum < 0.18, 0.18 + 0.35 * rgb, rgb)
    return np.clip(rgb, 0.0, 1.0)


def build_animation(tissue: HyphalTissue, rec: dict, outdir: Path) -> Path:
    xyz = tissue.xyz
    edges = list(tissue.G.edges)
    segs = np.array([[xyz[u], xyz[v]] for u, v in edges])
    eff = tissue.M * tissue.L
    ew = np.array([0.4 + 1.6 * (eff[u, v] - 1.0) / 4.0 for u, v in edges])
    ew = np.clip(ew, 0.3, 2.4)

    fig = plt.figure(figsize=(9.2, 7.2), facecolor="#0E1210")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0E1210")
    for spine in ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane:
        spine.set_facecolor((0.06, 0.08, 0.07, 1.0))
        spine.set_edgecolor((0.18, 0.22, 0.20, 0.4))
    ax.tick_params(colors="#6E7A72", labelsize=7)
    ax.set_xlabel("x", color="#8A948C")
    ax.set_ylabel("y", color="#8A948C")
    ax.set_zlabel("z", color="#8A948C")

    coll = Line3DCollection(segs, colors="#3A433C", linewidths=ew, alpha=0.55)
    ax.add_collection3d(coll)
    sizes0 = 28 + 90 * rec["gate"][0] / (rec["gate"].max() + 1e-9)
    sc = ax.scatter(
        xyz[:, 0], xyz[:, 1], xyz[:, 2],
        c=node_colors(rec["fe"][0], rec["ph"][0], rec["zn"][0]),
        s=sizes0, depthshade=False, edgecolors="#111", linewidths=0.3,
    )
    ln = xyz[tissue.is_ln]
    ax.scatter(ln[:, 0], ln[:, 1], ln[:, 2], facecolors="none",
               edgecolors="#C9A6E8", s=90, linewidths=1.0, label="Ln")
    src = xyz[tissue.source_fe]
    dst = xyz[tissue.dest]
    ax.scatter(*src, c="#E25B4A", s=70, marker="*")
    ax.scatter(*dst, c="#7DCEA0", s=50, marker="^")

    title = ax.set_title("", color="#D5DDD6", fontsize=11, pad=8)
    ax.legend(loc="upper left", fontsize=8, facecolor="#151A17",
              edgecolor="none", labelcolor="#C5CEC6")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_zlim(0, 1)

    nF = rec["fe"].shape[0]
    gmax = float(rec["gate"].max()) + 1e-9

    def update(fi: int):
        rgb = node_colors(rec["fe"][fi], rec["ph"][fi], rec["zn"][fi])
        sc._offsets3d = (xyz[:, 0], xyz[:, 1], xyz[:, 2])
        sc.set_color(rgb)
        sizes = 26 + 110 * rec["gate"][fi] / gmax
        sizes = np.where(rec["fired"][fi], sizes + 70, sizes)
        sc.set_sizes(sizes)
        ec = np.where(rec["fired"][fi][:, None], np.array([1.0, 0.95, 0.75]), np.array([0.07, 0.07, 0.07]))
        sc.set_edgecolors(ec)
        az = 28 + 0.35 * fi
        ax.view_init(elev=18, azim=az)
        t = rec["times"][fi]
        nfire = int(rec["fired"][fi].sum())
        title.set_text(
            f"ARCAM-Δχ   t={t:6.1f}   picos neste quadro={nfire}   "
            f"(vermelho=Fe  azul=P  ouro=Zn)"
        )
        return sc, title

    anim = FuncAnimation(fig, update, frames=nF, interval=70, blit=False)
    outdir.mkdir(parents=True, exist_ok=True)
    gif = outdir / "sim_hifal_anim.gif"
    anim.save(gif, writer=PillowWriter(fps=14), dpi=90)
    plt.close(fig)
    return gif


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Anima a malha hifal 3D.")
    ap.add_argument("--nodes", type=int, default=48)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "sim_hifal"),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    p = Params(
        n_nodes=args.nodes,
        steps=args.steps,
        seed=args.seed,
        sigma_chem=args.sigma,
        k_neighbors=4,
        anastomosis_extra=6,
    )
    print(f"simulando {p.n_nodes} nós × {p.steps} passos…")
    tissue = HyphalTissue(p)
    rec = record(tissue, args.stride)
    print(f"quadros={rec['fe'].shape[0]}  AND={len(tissue.and_events)}  "
          f"BLOCK={len(tissue.inh_events)}")
    outdir = Path(args.out)
    gif = build_animation(tissue, rec, outdir)
    print(f"gif: {gif}")


if __name__ == "__main__":
    main()
