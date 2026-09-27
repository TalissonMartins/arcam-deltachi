#!/usr/bin/env python3
"""
Ponto de entrada. Rode ESTE arquivo com Python:

    python simulador.py
    python simulador.py --animar
    python simulador.py --simular --nodes 64 --steps 2000

Não cole comandos de terminal aqui. Este arquivo já é o programa.
Precisa estar na mesma pasta que simular_rede_hifal.py e animar_rede_hifal.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description="ARCAM-Δχ — simulador da malha hifal")
    ap.add_argument("--animar", action="store_true", help="gera a animação 3D (padrão)")
    ap.add_argument("--simular", action="store_true", help="roda só a simulação e os gráficos")
    ap.add_argument("--nodes", type=int, default=48)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sigma", type=float, default=0.02)
    args, extra = ap.parse_known_args()

    missing = [n for n in ("simular_rede_hifal.py", "animar_rede_hifal.py") if not (ROOT / n).exists()]
    if missing:
        print("Arquivos ausentes nesta pasta:")
        for n in missing:
            print(f"  - {n}")
        print(f"Pasta atual: {ROOT}")
        print("Copie simular_rede_hifal.py e animar_rede_hifal.py para o mesmo diretório.")
        sys.exit(1)

    if args.simular and not args.animar:
        from simular_rede_hifal import Params, HyphalTissue, plot_results, summarize

        p = Params(n_nodes=args.nodes, steps=args.steps, seed=args.seed, sigma_chem=args.sigma)
        tissue = HyphalTissue(p)
        out = tissue.run()
        outdir = ROOT / "sim_hifal"
        plot_results(tissue, out, outdir)
        print(summarize(tissue, out))
        print(f"\nfiguras em {outdir}")
        return

    from animar_rede_hifal import record, build_animation
    from simular_rede_hifal import HyphalTissue, Params

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
    gif = build_animation(tissue, rec, ROOT / "sim_hifal")
    print(f"gif: {gif}")


if __name__ == "__main__":
    main()
