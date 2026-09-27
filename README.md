# ARCAM-Δχ

Malha micorrízica amazônica como computador neuromórfico **evento-dirigido**.

Porta química assimétrica em sítios de terra rara (Ln³⁺):

```
FIRE  ⇔  Fe ∧ P ∧ ¬Zn
```

Fosfato abre o sítio (Hill). Ferro injeta corrente só no sítio aberto. Zinco é ligante competitivo — a porta NOT. Três testemunhas espacialmente separadas votam 2-de-3; ruído local não grava LTP.

Isto **não** é um computador quântico. O parentesco com o programa topológico (Majorana) é arquitetural: a informação útil não mora num ponto; mora na relação entre pontos.

Autor: [Talisson Martins](https://github.com/TalissonMartins) · Cuiabá/MT

## Arquivos

| arquivo | função |
|---|---|
| `simulador.py` | ponto de entrada |
| `simular_rede_hifal.py` | motor: difusão 3D, porta Ln, STM/LTP, voto 2-de-3 |
| `animar_rede_hifal.py` | GIF 3D da pluma |

## Instalação

```bash
pip install numpy matplotlib networkx pillow
```

## Uso

```bash
python simulador.py
python simulador.py --animar --nodes 48 --steps 1600 --stride 8
python simulador.py --simular --nodes 64 --steps 2000
```

Windows: rode o comando no **Terminal**, na pasta dos três `.py`. Não cole `python …` dentro do arquivo.

Saída em `sim_hifal/`.

## Núcleo

```
I_Ln = G · Hill(P) · (1 − Hill(Zn)) · Fe^n
y    = 1[ voto_2/3 (Fe ∧ P ∧ ¬Zn) ]
```

Só `y` alimenta corrente, spike e potenciação de longo prazo.

## Licença

Pesquisa / arquitetura teórica. Use e cite. Não é produto nem organismo modificado.
