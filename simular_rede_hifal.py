#!/usr/bin/env python3
"""
ARCAM-Δχ — malha hifal como rede neuromórfica em grafo.

Três espécies iônicas no mesmo tecido:
  Fe  — ferro (χ_fe) — corrente
  P   — fosfato (χ_p) — abre o sítio
  Zn  — zinco (χ_zn) — ligante competitivo, negação

Nós Ln são portas químicas assimétricas, não amplificadores simétricos.
Fosfato abre o sítio de coordenação do Ln³⁺ (curva de Hill).
Ferro só injeta corrente nesse sítio já aberto — expoente diferente.
Hifa comum não inicia disparo iônico: só conduz cabo e sinapse.
Evento AND só nasce no transdutor mineral.
Arestas têm duas escalas de memória:
  M — resposta imediata, teto 3,2, decai rápido (STM)
  L — potenciação de longo prazo do bioma; só captura quando M
      permanece acima do limiar por tempo suficiente ou quando
      AND se repete no mesmo filamento.

Dependências: numpy, matplotlib, networkx
Uso:
  python simular_rede_hifal.py
  python simular_rede_hifal.py --nodes 80 --steps 2500 --seed 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


@dataclass
class Params:
    n_nodes: int = 80
    k_neighbors: int = 5
    anastomosis_extra: int = 8
    lambda_ion: float = 0.16
    lambda_elec: float = 0.48
    dt: float = 0.25
    steps: int = 4000
    seed: int = 11

    # química — ferro se liga mais; fosfato difunde mais
    D_fe: float = 0.55
    D_p: float = 0.95
    D_zn: float = 0.70
    decay_fe: float = 0.016
    decay_p: float = 0.009
    decay_zn: float = 0.012
    chi_th: float = 0.48
    chi_th_off: float = 0.30
    chi0: float = 0.08
    sigma_chem: float = 0.0   # ruído gaussiano em Fe e P (todos os nós)

    plume_amp_fe: float = 3.40
    plume_amp_p: float = 3.10
    plume_fe_start: int = 140
    plume_fe_duration: int = 520
    plume_p_start: int = 240
    plume_p_duration: int = 480
    plume_fe2_start: int = 1700
    plume_fe2_duration: int = 260
    plume_fe2_amp: float = 3.20
    # segunda coincidência, mais fraca: testa memória residual do filamento
    plume_fe3_start: int = 2300
    plume_fe3_duration: int = 280
    plume_fe3_amp: float = 2.10
    plume_p3_start: int = 2340
    plume_p3_duration: int = 260
    plume_p3_amp: float = 1.90
    # terceira coincidência, ainda mais fraca: L deve bastar onde M já esqueceu
    plume_fe4_start: int = 3300
    plume_fe4_duration: int = 240
    plume_fe4_amp: float = 1.55
    plume_p4_start: int = 3340
    plume_p4_duration: int = 220
    plume_p4_amp: float = 1.45
    # zinco no segundo overlap Fe∧P: bloqueia a porta (NOT)
    plume_zn_start: int = 2180
    plume_zn_duration: int = 520
    plume_zn_amp: float = 3.40

    # membrana / cabo
    C: float = 1.0
    g_L: float = 0.10
    E_L: float = -65.0
    g_gap: float = 0.22
    V_th: float = -50.0
    V_reset: float = -72.0
    t_ref: float = 6.0
    I_chem_gain: float = 0.0   # hifa comum: sem iniciação química
    G_ln: float = 48.0         # ganho da porta Ln só na coincidência
    K_p: float = 0.28          # fosfato que abre o sítio (Hill)
    n_p: float = 1.15          # Hill do fosfato — habilita
    n_fe: float = 2.05         # ferro — corrente assimétrica, mais íngreme
    K_zn: float = 0.32         # zinco que ocupa o sítio (Hill competitivo)
    n_zn: float = 1.80         # ocupação íngreme — inibição quase digital
    frac_ln: float = 0.10
    g_syn: float = 5.4
    used_window: float = 3.0

    # AND temporal Fe ∧ P no mesmo nó
    coincidence_window: float = 8.0

    # plasticidade iônica das arestas (memristência de curto prazo)
    eta_plast: float = 0.22
    tau_mem: float = 90.0
    m_max: float = 3.2
    spike_plast: float = 0.22
    # LTP só na aresta que conduziu spike — sem produto externo global
    theta_ltp: float = 2.05
    t_capture: float = 14.0
    eta_ltp: float = 0.18
    eta_and_ltp: float = 0.08
    tau_ltp: float = 700.0
    l_max: float = 2.6

    e_spike: float = 1.0
    e_event: float = 4.0
    e_poll_classic: float = 25.0


def build_hyphal_graph(p: Params, rng: np.random.Generator) -> tuple[nx.Graph, np.ndarray]:
    """Malha 3D densa: k-vizinhos euclidianos + anastomoses longas."""
    xyz = rng.random((p.n_nodes, 3))
    G = nx.Graph()
    G.add_nodes_from(range(p.n_nodes))

    for i in range(p.n_nodes):
        d2 = np.sum((xyz - xyz[i]) ** 2, axis=1)
        d2[i] = np.inf
        nn = np.argsort(d2)[: p.k_neighbors]
        for j in nn:
            length = float(np.sqrt(d2[j]))
            G.add_edge(int(i), int(j), length=max(length, 0.04))

    for _ in range(p.anastomosis_extra):
        a, b = rng.integers(0, p.n_nodes, size=2)
        if a != b and not G.has_edge(int(a), int(b)):
            length = float(np.linalg.norm(xyz[a] - xyz[b]))
            G.add_edge(int(a), int(b), length=max(length, 0.04))

    return G, xyz


def _plume(k: int, start: int, duration: int, amp: float) -> float:
    if start <= k < start + duration:
        phase = (k - start) / max(duration, 1)
        return amp * float(np.sin(np.pi * phase) ** 2)
    return 0.0


class HyphalTissue:
    def __init__(self, p: Params):
        self.p = p
        self.rng = np.random.default_rng(p.seed)
        self.G, self.xyz = build_hyphal_graph(p, self.rng)
        self.xy = self.xyz[:, :2]

        n = p.n_nodes
        self.fe = self.rng.normal(0.08, 0.02, size=n).clip(0.0, None)
        self.ph = self.rng.normal(0.08, 0.02, size=n).clip(0.0, None)
        self.zn = self.rng.normal(0.04, 0.01, size=n).clip(0.0, None)
        self.V = p.E_L + self.rng.normal(0.0, 1.2, size=n)
        self.ref = np.zeros(n)
        self.is_ln = np.zeros(n, dtype=bool)
        n_ln = max(2, int(p.frac_ln * n))
        self.is_ln[self.rng.choice(n, size=n_ln, replace=False)] = True

        # entrada num canto; destino no oposto — roteamento, não vizinhança
        self.source_fe = int(np.argmin(self.xyz[:, 0] + 0.15 * self.xyz[:, 1] + 0.10 * self.xyz[:, 2]))
        d2 = np.sum((self.xyz - self.xyz[self.source_fe]) ** 2, axis=1)
        d2[self.source_fe] = np.inf
        self.source_p = int(np.argsort(d2)[2])
        self.dest = int(np.argmax(np.sum((self.xyz - self.xyz[self.source_fe]) ** 2, axis=1)))
        self.source_fe2 = int(np.argmax(np.sum((self.xyz - self.xyz[self.source_p]) ** 2, axis=1)))
        if self.source_fe2 == self.dest:
            order = np.argsort(np.sum((self.xyz - self.xyz[self.source_p]) ** 2, axis=1))
            self.source_fe2 = int(order[-2])
        self.source_zn = self.source_p
        self.is_ln[self.source_p] = True
        self.is_ln[self.source_fe] = True
        self.is_ln[self.dest] = True
        self.detour = self._carve_detour()
        self.neighbors = [list(self.G.neighbors(i)) for i in range(n)]

        self.Len = np.zeros((n, n))
        W_ion = np.zeros((n, n))
        W_el = np.zeros((n, n))
        for u, v, data in self.G.edges(data=True):
            Lij = max(float(data["length"]), 0.04)
            self.Len[u, v] = self.Len[v, u] = Lij
            # íon atenua forte com a distância; spike elétrico vai mais longe
            a_ion = float(np.exp(-Lij / max(p.lambda_ion, 1e-6)))
            a_el = float(np.exp(-Lij / max(p.lambda_elec, 1e-6)))
            W_ion[u, v] = W_ion[v, u] = a_ion / Lij
            W_el[u, v] = W_el[v, u] = a_el / Lij
        self.W0_ion = W_ion
        self.W0_el = W_el
        self.W0 = W_el.copy()
        self.M = np.ones((n, n))
        self.L = np.ones((n, n))
        self.dwell = np.zeros((n, n))
        self.W_hat = np.zeros((n, n))
        self.W_ion_hat = np.zeros((n, n))
        self.W_el_hat = np.zeros((n, n))
        self.I_gate = np.zeros(n)
        self.mean_m = 1.0
        self.max_m = 1.0
        self.mean_l = 1.0
        self.max_l = 1.0
        self.max_eff = 1.0
        self._rebuild_what()

        self.spikes: list[tuple[float, int]] = []
        self.spikes_ln: list[tuple[float, int]] = []
        self.peaks_fe: list[tuple[float, int, float]] = []
        self.peaks_p: list[tuple[float, int, float]] = []
        self.and_events: list[tuple[float, int, float, float]] = []
        self.inh_events: list[tuple[float, int, float, float, float]] = []

        self.armed_fe = np.ones(n, dtype=bool)
        self.armed_p = np.ones(n, dtype=bool)
        self.armed_and = np.ones(n, dtype=bool)
        self.armed_inh = np.ones(n, dtype=bool)
        self.occ_zn = np.zeros(n)
        self.last_fe_t = np.full(n, -1e9)
        self.last_p_t = np.full(n, -1e9)
        self.last_and_t = np.full(n, -1e9)
        self.last_spike_t = np.full(n, -1e9)
        self.last_tx = np.full((n, n), -1e9)
        self.I_syn = np.zeros(n)
        self.n_spikes = 0
        self.n_polls_classic = 0

    def _carve_detour(self) -> list[int]:
        """Corredor longo, deslocado da geodésica, com portas Ln — alvo da LTP."""
        src, dst = self.source_fe, self.dest
        axis = self.xyz[dst] - self.xyz[src]
        nrm = float(np.linalg.norm(axis)) + 1e-9
        u = axis / nrm
        ref = np.array([0.0, 1.0, 0.0]) if abs(u[1]) < 0.85 else np.array([0.0, 0.0, 1.0])
        v = ref - u * float(ref @ u)
        v = v / (float(np.linalg.norm(v)) + 1e-9)
        tproj = (self.xyz - self.xyz[src]) @ u
        off = (self.xyz - self.xyz[src]) @ v
        band = (tproj > 0.06 * nrm) & (tproj < 0.94 * nrm) & (off > 0.10)
        cand = np.flatnonzero(band)
        if cand.size < 4:
            band = (tproj > 0.06 * nrm) & (tproj < 0.94 * nrm) & (np.abs(off) > 0.05)
            cand = np.flatnonzero(band)
        if cand.size < 3:
            return [src, dst]
        order = cand[np.argsort(tproj[cand])]
        pick = np.linspace(0, order.size - 1, min(6, order.size)).astype(int)
        way = [src] + [int(order[i]) for i in pick] + [dst]
        seen: list[int] = []
        for node in way:
            if node not in seen:
                seen.append(node)
        way = seen
        for a, b in zip(way[:-1], way[1:]):
            if a == b:
                continue
            if not self.G.has_edge(a, b):
                length = float(np.linalg.norm(self.xyz[a] - self.xyz[b]))
                self.G.add_edge(a, b, length=max(length, 0.04))
            self.is_ln[a] = True
            self.is_ln[b] = True
        return way

    def _norm(self, W: np.ndarray) -> np.ndarray:
        deg = W.sum(axis=1, keepdims=True)
        deg[deg == 0.0] = 1.0
        return W / deg

    def _rebuild_what(self) -> None:
        plast = self.M * self.L
        self.W_ion_hat = self._norm(self.W0_ion * plast)
        self.W_el_hat = self._norm(self.W0_el * plast)
        self.W_hat = self.W_el_hat
        mask = self.W0 > 0
        if np.any(mask):
            self.mean_m = float(self.M[mask].mean())
            self.max_m = float(self.M[mask].max())
            self.mean_l = float(self.L[mask].mean())
            self.max_l = float(self.L[mask].max())
            self.max_eff = float((self.M[mask] * self.L[mask]).max())

    def _plasticity(self, t: float, xf: np.ndarray, xp: np.ndarray) -> None:
        """STM/LTP só na aresta que acabou de transmitir — o resto do grafo não aprende."""
        p = self.p
        edges = self.W0 > 0
        used = ((t - self.last_tx) <= p.used_window) | ((t - self.last_tx.T) <= p.used_window)
        ln_touch = self.is_ln[:, None] | self.is_ln[None, :]
        used = used & edges & ln_touch
        load = np.sqrt(np.maximum(xf * xp, 0.0))
        hebb = np.outer(load, load) * used
        grow = p.eta_plast * hebb + p.spike_plast * used.astype(float)
        np.fill_diagonal(grow, 0.0)
        self.M = self.M + p.dt * grow
        decay_m = np.exp(-p.dt / max(p.tau_mem, 1e-6))
        self.M = 1.0 + (self.M - 1.0) * decay_m
        self.M = np.clip(self.M, 1.0, p.m_max)
        self.M = np.maximum(self.M, self.M.T)

        tagged = (self.M >= p.theta_ltp) & used
        self.dwell = np.where(tagged, self.dwell + p.dt, self.dwell * np.exp(-p.dt / 3.0))
        capture = (self.dwell >= p.t_capture) & used
        and_now = ((t - self.last_and_t) <= p.used_window)
        and_edge = used & and_now[:, None] & and_now[None, :]
        dL = p.eta_ltp * (self.M - 1.0) * capture + p.eta_and_ltp * and_edge.astype(float)
        np.fill_diagonal(dL, 0.0)
        self.L = self.L + p.dt * dL
        decay_l = np.exp(-p.dt / max(p.tau_ltp, 1e-6))
        self.L = 1.0 + (self.L - 1.0) * decay_l
        self.L = np.clip(self.L, 1.0, p.l_max)
        self.L = np.maximum(self.L, self.L.T)
        self._rebuild_what()

    def _diffuse(self, field: np.ndarray, D: float, decay: float) -> np.ndarray:
        return D * (self.W_ion_hat @ field - field) - decay * field

    def _register_and(self, t: float, i: int) -> None:
        """AND só se Fe∧P no Ln e o sítio não estiver ocupado por Zn."""
        p = self.p
        both_high = (self.fe[i] >= p.chi_th) and (self.ph[i] >= p.chi_th)
        peaks_close = abs(self.last_fe_t[i] - self.last_p_t[i]) <= p.coincidence_window
        if not (both_high or peaks_close):
            return
        if not self.is_ln[i]:
            return
        blocked = self.zn[i] >= p.chi_th or self.occ_zn[i] >= 0.45
        if blocked:
            if self.armed_inh[i]:
                self.armed_inh[i] = False
                self.inh_events.append(
                    (t, int(i), float(self.fe[i]), float(self.ph[i]), float(self.zn[i]))
                )
            return
        if not self.armed_and[i]:
            return
        self.armed_and[i] = False
        self.last_and_t[i] = t
        self.and_events.append((t, int(i), float(self.fe[i]), float(self.ph[i])))

    def step(self, t: float, k: int) -> None:
        p = self.p
        fe, ph, V = self.fe, self.ph, self.V

        inj_fe = _plume(k, p.plume_fe_start, p.plume_fe_duration, p.plume_amp_fe)
        inj_p = _plume(k, p.plume_p_start, p.plume_p_duration, p.plume_amp_p)
        inj_fe2 = _plume(k, p.plume_fe2_start, p.plume_fe2_duration, p.plume_fe2_amp)
        inj_fe3 = _plume(k, p.plume_fe3_start, p.plume_fe3_duration, p.plume_fe3_amp)
        inj_p3 = _plume(k, p.plume_p3_start, p.plume_p3_duration, p.plume_p3_amp)
        inj_fe4 = _plume(k, p.plume_fe4_start, p.plume_fe4_duration, p.plume_fe4_amp)
        inj_p4 = _plume(k, p.plume_p4_start, p.plume_p4_duration, p.plume_p4_amp)

        inj_zn = _plume(k, p.plume_zn_start, p.plume_zn_duration, p.plume_zn_amp)
        zn = self.zn

        dfe = self._diffuse(fe, p.D_fe, p.decay_fe)
        dph = self._diffuse(ph, p.D_p, p.decay_p)
        dzn = self._diffuse(zn, p.D_zn, p.decay_zn)
        dfe[self.source_fe] += inj_fe + inj_fe3 + inj_fe4
        dph[self.source_p] += inj_p + inj_p3 + inj_p4
        dfe[self.source_fe2] += inj_fe2
        dzn[self.source_zn] += inj_zn
        fe[:] = np.clip(fe + p.dt * dfe, 0.0, 3.5)
        ph[:] = np.clip(ph + p.dt * dph, 0.0, 3.5)
        zn[:] = np.clip(zn + p.dt * dzn, 0.0, 3.5)
        if p.sigma_chem > 0.0:
            n = fe.size
            fe[:] = np.clip(fe + self.rng.normal(0.0, p.sigma_chem, size=n), 0.0, 3.5)
            ph[:] = np.clip(ph + self.rng.normal(0.0, p.sigma_chem, size=n), 0.0, 3.5)

        I_gap = p.g_gap * (self.W_hat @ V - V)
        leak = -p.g_L * (V - p.E_L)
        xf = np.maximum(fe - p.chi0, 0.0)
        xp = np.maximum(ph - p.chi0, 0.0)
        xz = np.maximum(zn - p.chi0, 0.0)
        open_site = (xp ** p.n_p) / (xp ** p.n_p + p.K_p ** p.n_p + 1e-12)
        self.occ_zn = (xz ** p.n_zn) / (xz ** p.n_zn + p.K_zn ** p.n_zn + 1e-12)
        I_ln = p.G_ln * open_site * (1.0 - self.occ_zn) * (xf ** p.n_fe)
        self.I_gate = np.where(self.is_ln, I_ln, 0.0)
        dV = (leak + I_gap + self.I_gate + self.I_syn) / p.C
        self.I_syn *= 0.55

        active = self.ref <= 0.0
        V[active] = V[active] + p.dt * dV[active]
        self.ref = np.maximum(0.0, self.ref - p.dt)

        fired = (V >= p.V_th) & active
        if np.any(fired):
            idx = np.flatnonzero(fired)
            V[fired] = p.V_reset
            self.ref[fired] = p.t_ref
            self.n_spikes += int(idx.size)
            for i in idx:
                self.spikes.append((t, int(i)))
                self.last_spike_t[i] = t
                if self.is_ln[i] and self.I_gate[i] > 0.4:
                    self.spikes_ln.append((t, int(i)))
                for j in self.neighbors[i]:
                    att = float(np.exp(-self.Len[i, j] / max(p.lambda_elec, 1e-6)))
                    self.I_syn[j] += p.g_syn * float(self.M[i, j] * self.L[i, j]) * att
                    self.last_tx[i, j] = t

        # picos de cada espécie (histerese) + filtro AND no mesmo nó
        rise_fe = self.armed_fe & (fe >= p.chi_th)
        if np.any(rise_fe):
            for i in np.flatnonzero(rise_fe):
                self.peaks_fe.append((t, int(i), float(fe[i])))
                self.last_fe_t[i] = t
                self._register_and(t, int(i))
            self.armed_fe[rise_fe] = False
        self.armed_fe[fe <= p.chi_th_off] = True

        rise_p = self.armed_p & (ph >= p.chi_th)
        if np.any(rise_p):
            for i in np.flatnonzero(rise_p):
                self.peaks_p.append((t, int(i), float(ph[i])))
                self.last_p_t[i] = t
                self._register_and(t, int(i))
            self.armed_p[rise_p] = False
        self.armed_p[ph <= p.chi_th_off] = True
        # copresenca: Fe e P altos agora, mesmo que os flancos nao tenham coincidido
        both_now = self.armed_and & (fe >= p.chi_th) & (ph >= p.chi_th)
        if np.any(both_now):
            for i in np.flatnonzero(both_now):
                self._register_and(t, int(i))
        self.armed_and[(fe <= p.chi_th_off) | (ph <= p.chi_th_off)] = True
        self.armed_inh[zn <= p.chi_th_off] = True

        self._plasticity(t, xf, xp)
        self.n_polls_classic += p.n_nodes

    def run(self) -> dict:
        p = self.p
        trace_fe = np.zeros(p.steps)
        trace_p = np.zeros(p.steps)
        trace_zn = np.zeros(p.steps)
        trace_v = np.zeros(p.steps)
        trace_gate = np.zeros(p.steps)
        trace_occ = np.zeros(p.steps)
        trace_state = np.zeros(p.steps, dtype=int)
        trace_open = np.zeros(p.steps)
        trace_block = np.zeros(p.steps)
        trace_m = np.zeros(p.steps)
        trace_l = np.zeros(p.steps)
        n_ln = max(1, int(self.is_ln.sum()))

        for k in range(p.steps):
            t = k * p.dt
            self.step(t, k)
            ln = self.is_ln
            trace_fe[k] = float(self.fe[ln].mean()) if n_ln else float(self.fe.mean())
            trace_p[k] = float(self.ph[ln].mean()) if n_ln else float(self.ph.mean())
            trace_zn[k] = float(self.zn[ln].mean()) if n_ln else float(self.zn.mean())
            trace_v[k] = float(self.V[ln].mean()) if n_ln else float(self.V.mean())
            trace_gate[k] = float(self.I_gate[ln].mean()) if n_ln else 0.0
            trace_occ[k] = float(self.occ_zn[ln].mean()) if n_ln else 0.0
            fe_hi = trace_fe[k] >= p.chi_th
            p_hi = trace_p[k] >= p.chi_th
            z_hi = trace_occ[k] >= 0.45 or trace_zn[k] >= p.chi_th
            if z_hi and fe_hi and p_hi:
                trace_state[k] = 4  # BLOCK  Fe∧P∧Zn
            elif fe_hi and p_hi:
                trace_state[k] = 3  # FIRE   Fe∧P∧¬Zn
            elif z_hi:
                trace_state[k] = 5  # INHIB  Zn
            elif p_hi:
                trace_state[k] = 1  # OPEN   P
            elif fe_hi:
                trace_state[k] = 2  # DRIVE  Fe
            else:
                trace_state[k] = 0  # IDLE
            trace_open[k] = 1.0 if trace_state[k] == 3 else 0.0
            trace_block[k] = 1.0 if trace_state[k] in (4, 5) else 0.0
            trace_m[k] = self.mean_m
            trace_l[k] = self.mean_l

        e_arcam = self.n_spikes * p.e_spike + len(self.and_events) * p.e_event
        e_classic = self.n_polls_classic * p.e_poll_classic
        rejected = len(self.peaks_fe) + len(self.peaks_p) - 2 * len(self.and_events)
        return {
            "trace_fe": trace_fe,
            "trace_p": trace_p,
            "trace_zn": trace_zn,
            "trace_occ": trace_occ,
            "trace_state": trace_state,
            "trace_open": trace_open,
            "trace_block": trace_block,
            "trace_v": trace_v,
            "trace_gate": trace_gate,
            "trace_m": trace_m,
            "trace_l": trace_l,
            "max_m": self.max_m,
            "max_l": self.max_l,
            "max_eff": self.max_eff,
            "e_arcam": e_arcam,
            "e_classic": e_classic,
            "gain": (e_classic / e_arcam) if e_arcam > 0 else np.inf,
            "rejected": max(0, rejected),
            "route": analyze_routing(self),
            "noise": classify_events(self),
        }


def _windows(p: Params) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    zn0 = p.plume_zn_start * p.dt
    zn1 = (p.plume_zn_start + p.plume_zn_duration) * p.dt
    pairs = [
        (p.plume_fe_start, p.plume_fe_duration, p.plume_p_start, p.plume_p_duration),
        (p.plume_fe3_start, p.plume_fe3_duration, p.plume_p3_start, p.plume_p3_duration),
        (p.plume_fe4_start, p.plume_fe4_duration, p.plume_p4_start, p.plume_p4_duration),
    ]
    true_w: list[tuple[float, float]] = []
    block_w: list[tuple[float, float]] = []
    for fs, fd, ps, pd in pairs:
        a = max(fs, ps) * p.dt
        b = min(fs + fd, ps + pd) * p.dt
        if a >= b:
            continue
        if b <= zn0 or a >= zn1:
            true_w.append((a, b))
            continue
        if a < zn0:
            true_w.append((a, min(b, zn0)))
        if b > zn1:
            true_w.append((max(a, zn1), b))
        lo, hi = max(a, zn0), min(b, zn1)
        if lo < hi:
            block_w.append((lo, hi))
    return true_w, block_w


def classify_events(tissue: HyphalTissue) -> dict:
    true_w, _block_w = _windows(tissue.p)

    def inside(t: float, wins: list[tuple[float, float]]) -> bool:
        return any(a <= t <= b for a, b in wins)

    fire_t = [t for t, *_ in tissue.and_events]
    n_true = sum(1 for t in fire_t if inside(t, true_w))
    n_false = len(fire_t) - n_true
    return {
        "sigma": tissue.p.sigma_chem,
        "fire_true": n_true,
        "fire_false": n_false,
        "block": len(tissue.inh_events),
        "mean_m": tissue.mean_m,
        "max_m": tissue.max_m,
        "mean_l": tissue.mean_l,
        "max_l": tissue.max_l,
        "spikes_ln": len(tissue.spikes_ln),
    }


def analyze_routing(tissue: HyphalTissue) -> dict:
    """Compara geodésica espacial e caminho de menor custo 1/(M L)."""
    src, dst = tissue.source_fe, tissue.dest
    H = nx.Graph()
    for u, v in tissue.G.edges:
        strength = float(tissue.M[u, v] * tissue.L[u, v])
        H.add_edge(u, v, cost=1.0 / max(strength, 1e-6), strength=strength)
    geo = nx.shortest_path(tissue.G, src, dst, weight="length")
    try:
        ltp = nx.shortest_path(H, src, dst, weight="cost")
    except nx.NetworkXNoPath:
        ltp = geo

    def mean_L(path: list[int]) -> float:
        if len(path) < 2:
            return 1.0
        vals = [float(tissue.L[path[i], path[i + 1]]) for i in range(len(path) - 1)]
        return float(np.mean(vals))

    def length(path: list[int]) -> float:
        if len(path) < 2:
            return 0.0
        return float(sum(tissue.G[path[i]][path[i + 1]]["length"] for i in range(len(path) - 1)))

    dest_spk = [t for t, i in tissue.spikes if i == dst]
    src_spk = [t for t, i in tissue.spikes if i == src]
    delay = None
    if dest_spk and src_spk:
        delay = float(dest_spk[0] - src_spk[0])
    return {
        "src": src,
        "dst": dst,
        "geo": geo,
        "ltp": ltp,
        "hops_geo": len(geo) - 1,
        "hops_ltp": len(ltp) - 1,
        "len_geo": length(geo),
        "len_ltp": length(ltp),
        "L_geo": mean_L(geo),
        "L_ltp": mean_L(ltp),
        "overlap": len(set(geo) & set(ltp)),
        "n_dest_spikes": len(dest_spk),
        "delay": delay,
        "chi_dest": (float(tissue.fe[dst]), float(tissue.ph[dst])),
        "dist3d": float(np.linalg.norm(tissue.xyz[src] - tissue.xyz[dst])),
        "n_strong": int(((tissue.L > 1.25) & (tissue.W0 > 0)).sum() // 2),
        "n_edges": int(tissue.G.number_of_edges()),
        "same_path": geo == ltp,
        "detour": list(getattr(tissue, "detour", [])),
    }


def plot_results(tissue: HyphalTissue, out: dict, outdir: Path) -> None:
    p = tissue.p
    t = np.arange(p.steps) * p.dt
    outdir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(11.4, 8.8), facecolor="white")
    fig.suptitle(
        "ARCAM-Δχ  ·  porta Fe∧P∧¬Zn (inibição competitiva)",
        fontsize=13,
        fontweight="bold",
        color="#1B4D3E",
    )

    ax1 = fig.add_subplot(2, 2, 1)
    pos = {i: tissue.xy[i] for i in range(p.n_nodes)}
    eff = tissue.M * tissue.L
    widths = [0.35 + 1.4 * (eff[u, v] - 1.0) / max(p.m_max * p.l_max - 1.0, 1e-6) for u, v in tissue.G.edges]
    nx.draw_networkx_edges(tissue.G, pos, ax=ax1, width=widths, edge_color="#C4BFB4", alpha=0.45)
    route = out["route"]
    geo_edges = list(zip(route["geo"][:-1], route["geo"][1:]))
    ltp_edges = list(zip(route["ltp"][:-1], route["ltp"][1:]))
    nx.draw_networkx_edges(
        tissue.G, pos, ax=ax1, edgelist=geo_edges,
        width=1.4, edge_color="#B33A2B", style="dashed", alpha=0.85,
    )
    nx.draw_networkx_edges(
        tissue.G, pos, ax=ax1, edgelist=ltp_edges,
        width=2.2, edge_color="#1B4D3E", alpha=0.95,
    )
    ax1.scatter(tissue.xy[:, 0], tissue.xy[:, 1], c=tissue.xyz[:, 2],
                cmap="YlOrBr", s=22, zorder=3, edgecolors="#1A1A1A", linewidths=0.2)
    ln_xy = tissue.xy[tissue.is_ln]
    ax1.scatter(ln_xy[:, 0], ln_xy[:, 1], facecolors="none", edgecolors="#7A1FA2",
                s=80, linewidths=1.2, zorder=4, label="Ln")
    sfe, sp, sd = tissue.xy[tissue.source_fe], tissue.xy[tissue.source_p], tissue.xy[tissue.dest]
    ax1.scatter([sfe[0]], [sfe[1]], c="#B33A2B", s=110, marker="*", zorder=5, label="entrada Fe")
    ax1.scatter([sp[0]], [sp[1]], c="#2B6B9A", s=70, marker="D", zorder=5, label="entrada P")
    ax1.scatter([sd[0]], [sd[1]], c="#1B4D3E", s=90, marker="^", zorder=5, label="destino")
    szn = tissue.xy[tissue.source_zn]
    ax1.scatter([szn[0]], [szn[1]], c="#C9A227", s=70, marker="P", zorder=5, label="entrada Zn")
    ax1.set_title("XY — verde LTP · tracejado geodesica")
    ax1.set_axis_off()
    ax1.legend(loc="upper right", fontsize=7, frameon=False)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(t, out["trace_fe"], color="#B33A2B", lw=1.6, label="Fe Ln")
    ax2.plot(t, out["trace_p"], color="#2B6B9A", lw=1.6, label="P Ln")
    ax2.plot(t, out["trace_zn"], color="#C9A227", lw=1.6, label="Zn Ln")
    ax2.axhline(p.chi_th, color="#5C5C5C", ls="--", lw=1, label="limiar")
    ax2.set_ylabel("concentracao")
    ax2.set_xlabel("tempo")
    ax2.set_title("plumas e corrente — Zn fecha a porta")
    ax2b = ax2.twinx()
    ax2b.plot(t, out["trace_gate"], color="#7A1FA2", lw=1.3, alpha=0.85, label="I_gate Ln")
    ax2b.set_ylabel("I_gate (u.a.)")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, fontsize=7.5, loc="upper left", frameon=False)

    ax3 = fig.add_subplot(2, 2, 3)
    if tissue.spikes:
        ts, ids = zip(*tissue.spikes)
        ax3.scatter(ts, ids, s=7, c="#B0B0B0", marker="|", linewidths=0.9, label="spike V")
    if tissue.peaks_fe:
        tf, iff, _ = zip(*tissue.peaks_fe)
        ax3.scatter(tf, iff, s=26, c="#B33A2B", marker="o", alpha=0.7, label="pico Fe")
    if tissue.peaks_p:
        tp, ip, _ = zip(*tissue.peaks_p)
        ax3.scatter(tp, ip, s=26, c="#2B6B9A", marker="s", alpha=0.7, label="pico P")
    if tissue.and_events:
        ta, ia, _, _ = zip(*tissue.and_events)
        ax3.scatter(ta, ia, s=64, facecolors="none", edgecolors="#1B4D3E",
                    linewidths=1.6, marker="o", label="AND Fe∧P∧¬Zn", zorder=5)
    if tissue.inh_events:
        ti, ii, *_ = zip(*tissue.inh_events)
        ax3.scatter(ti, ii, s=52, c="#C9A227", marker="x", linewidths=1.4,
                    label="INH Fe∧P∧Zn", zorder=6)
    ax3.set_xlabel("tempo")
    ax3.set_ylabel("no")
    ax3.set_title("FIRE vs NOT no transdutor Ln")
    ax3.set_ylim(-1, p.n_nodes)
    ax3.legend(fontsize=7.5, loc="upper right", frameon=False)

    ax4 = fig.add_subplot(2, 2, 4)
    labels_s = ["IDLE", "OPEN P", "DRIVE Fe", "FIRE", "BLOCK", "INHIB"]
    ax4.step(t, out["trace_state"], where="post", color="#1B4D3E", lw=1.6)
    ax4.set_yticks(range(6))
    ax4.set_yticklabels(labels_s, fontsize=7.5)
    ax4.set_xlabel("tempo")
    ax4.set_title("transicoes de estado da porta Ln")
    ax4.set_ylim(-0.3, 5.3)
    ax4.text(
        0.5, 0.04,
        f"FIRE={len(tissue.and_events)}  BLOCK={len(tissue.inh_events)}  "
        f"I = G·Hill(P)·(1-Hill(Zn))·Fe^n",
        transform=ax4.transAxes, ha="center", fontsize=8, color="#5C5C5C",
    )

    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(outdir / "sim_hifal_arcam.png", dpi=140)
    plt.close(fig)

    figp, (axp, axg) = plt.subplots(
        2, 1, figsize=(10.6, 6.4), sharex=True,
        gridspec_kw={"height_ratios": [1.35, 0.75]},
    )
    figp.suptitle("porta inteligente  ·  abre em Fe∧P  ·  fecha com Zn em excesso",
                  fontsize=12, fontweight="bold", color="#1B4D3E")
    axp.plot(t, out["trace_fe"], color="#B33A2B", lw=1.7, label="Fe")
    axp.plot(t, out["trace_p"], color="#2B6B9A", lw=1.7, label="P")
    axp.plot(t, out["trace_zn"], color="#C9A227", lw=1.8, label="Zn")
    axp.axhline(p.chi_th, color="#5C5C5C", ls="--", lw=1)
    axp.set_ylabel("concentração nos Ln")
    axpb = axp.twinx()
    axpb.plot(t, out["trace_gate"], color="#7A1FA2", lw=1.4, alpha=0.9, label="I_gate")
    axpb.set_ylabel("I_gate")
    h1, l1 = axp.get_legend_handles_labels()
    h2, l2 = axpb.get_legend_handles_labels()
    axp.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", frameon=False)
    axp.set_title("substratos e corrente")

    axg.fill_between(t, 0, out["trace_open"], color="#1B4D3E", alpha=0.55, label="ABERTA  Fe∧P∧¬Zn")
    axg.fill_between(t, 0, out["trace_block"], color="#C9A227", alpha=0.70, label="BLOQUEADA  Zn")
    axg.set_ylim(0, 1.35)
    axg.set_yticks([0, 1])
    axg.set_yticklabels(["fechada", "ativa"])
    axg.set_xlabel("tempo")
    axg.set_title("estado da porta")
    axg.legend(fontsize=8, loc="upper right", frameon=False)
    figp.tight_layout(rect=[0, 0.02, 1, 0.95])
    figp.savefig(outdir / "sim_hifal_porta.png", dpi=140)
    plt.close(figp)

    fig2, ax = plt.subplots(figsize=(10.2, 4.4))
    if tissue.spikes:
        ts, ids = zip(*tissue.spikes)
        colors = ["#7A1FA2" if tissue.is_ln[i] else "#888888" for i in ids]
        ax.scatter(ts, ids, s=9, c=colors, marker="|", linewidths=1.1)
    if tissue.and_events:
        ta, ia, _, _ = zip(*tissue.and_events)
        ax.scatter(ta, ia, s=70, facecolors="none", edgecolors="#1B4D3E",
                   linewidths=1.6, marker="o", label="AND Fe∧P", zorder=5)
    ax.set_xlabel("tempo")
    ax.set_ylabel("indice do no")
    ax.set_title("raster — cinza/violeta: spike · circulo: assinatura Fe∧P")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig2.tight_layout()
    fig2.savefig(outdir / "sim_hifal_raster.png", dpi=140)
    plt.close(fig2)

    fig3 = plt.figure(figsize=(8.4, 6.6))
    ax = fig3.add_subplot(111, projection="3d")
    for u, v in tissue.G.edges:
        a, b = tissue.xyz[u], tissue.xyz[v]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#D0CBC2", lw=0.4, alpha=0.35)
    rp = out["route"]["ltp"]
    for i in range(len(rp) - 1):
        a, b = tissue.xyz[rp[i]], tissue.xyz[rp[i + 1]]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color="#1B4D3E", lw=2.0)
    ax.scatter(tissue.xyz[:, 0], tissue.xyz[:, 1], tissue.xyz[:, 2],
               c=tissue.xyz[:, 2], cmap="YlOrBr", s=12, depthshade=True)
    ax.scatter(*tissue.xyz[tissue.source_fe], c="#B33A2B", s=60, marker="*")
    ax.scatter(*tissue.xyz[tissue.dest], c="#1B4D3E", s=50, marker="^")
    ax.set_title("malha 3D — rota LTP fonte→destino")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    fig3.tight_layout()
    fig3.savefig(outdir / "sim_hifal_3d.png", dpi=140)
    plt.close(fig3)


def summarize(tissue: HyphalTissue, out: dict) -> str:
    p = tissue.p
    lines = [
        "=== ARCAM-Δχ  malha 3D + roteamento LTP ===",
        f"nos={p.n_nodes}  arestas={tissue.G.number_of_edges()}  dim=3  "
        f"Ln={int(tissue.is_ln.sum())}",
        f"lambda_ion={p.lambda_ion}  lambda_elec={p.lambda_elec}",
        f"entrada Fe={tissue.source_fe}  entrada P={tissue.source_p}  destino={tissue.dest}",
        f"janela AND         : {p.coincidence_window} (somente no Ln)",
        f"spikes de membrana : {tissue.n_spikes}",
        f"spikes iniciados Ln: {len(tissue.spikes_ln)}  (porta aberta)",
        f"picos de ferro     : {len(tissue.peaks_fe)}",
        f"picos de fosfato   : {len(tissue.peaks_p)}",
        f"porta Ln           : I = G·Hill(P)·(1-Hill(Zn))·Fe^n   NOT competitivo",
        f"eventos FIRE Fe∧P∧¬Zn : {len(tissue.and_events)}",
        f"eventos BLOCK Fe∧P∧Zn : {len(tissue.inh_events)}  ← inibicao",
        f"ruido sigma        : {p.sigma_chem:.3f}  FIRE verdadeiro={out.get('noise',{}).get('fire_true','?')}  "
        f"falso={out.get('noise',{}).get('fire_false','?')}",
        f"STM  M             : medio={tissue.mean_m:.3f}  max={out['max_m']:.3f}  tau={p.tau_mem}  teto={p.m_max}",
        f"LTP  L             : medio={tissue.mean_l:.3f}  max={out['max_l']:.3f}  tau={p.tau_ltp}  teto={p.l_max}",
        f"acoplamento M x L  : max efetivo={out['max_eff']:.3f}  (supera o teto 3.2 da STM)",
        f"picos rejeitados   : ≈ {out['rejected']}  (especie isolada)",
        f"custo ARCAM        : {out['e_arcam']:.1f} u.a.",
        f"custo classico     : {out['e_classic']:.1f} u.a.",
        f"ganho energetico   : ≈ {out['gain']:.0f}×",
    ]
    if tissue.and_events:
        t0, i0, fe0, p0 = tissue.and_events[0]
        lines.append(f"primeiro AND       : t={t0:.1f}  no={i0}  Fe={fe0:.2f}  P={p0:.2f}")
    r = out.get("route") or {}
    if r:
        delay = f"{r['delay']:.1f}" if r.get("delay") is not None else "n/a"
        lines += [
            f"distancia 3D       : {r['dist3d']:.3f}",
            f"geodesica          : {r['hops_geo']} hops  Lmedio={r['L_geo']:.3f}  len={r['len_geo']:.3f}",
            f"rota LTP           : {r['hops_ltp']} hops  Lmedio={r['L_ltp']:.3f}  len={r['len_ltp']:.3f}",
            f"intersecao nos     : {r['overlap']}  spikes no destino={r['n_dest_spikes']}  atraso={delay}",
            f"chi no destino     : Fe={r['chi_dest'][0]:.3f}  P={r['chi_dest'][1]:.3f}  (atenuacao ionica)",
            f"arestas L>1.25     : {r.get('n_strong', 0)}/{r.get('n_edges', 0)}  "
            f"LTP≠geo={not r.get('same_path', True)}",
        ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Simula malha hifal com filtro AND Fe∧P.")
    ap.add_argument("--nodes", type=int, default=80)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--sigma", type=float, default=0.10)
    ap.add_argument("--sweep", action="store_true", default=True)
    ap.add_argument(
        "--out",
        type=str,
        default=str(Path(__file__).resolve().parent / "sim_hifal"),
    )
    return ap.parse_args()


def plot_noise_sweep(rows: list[dict], outdir: Path) -> None:
    sig = np.array([r["sigma"] for r in rows])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.2))
    fig.suptitle("tolerância ao ruído químico gaussiano", fontsize=12,
                 fontweight="bold", color="#1B4D3E")
    ax1.plot(sig, [r["fire_true"] for r in rows], "o-", color="#1B4D3E", label="FIRE verdadeiro")
    ax1.plot(sig, [r["fire_false"] for r in rows], "s-", color="#B33A2B", label="FIRE falso")
    ax1.axhline(1.0, color="#A3A095", ls="--", lw=0.8)
    ax1.set_xlabel("σ ruído (Fe, P)")
    ax1.set_ylabel("eventos")
    ax1.set_title("limiar de falso positivo")
    ax1.legend(fontsize=8, frameon=False)
    ax2.plot(sig, [r["max_m"] for r in rows], "o-", color="#1B4D3E", label="M max STM")
    ax2.plot(sig, [r["max_l"] for r in rows], "s-", color="#7A1FA2", label="L max LTP")
    ax2.axhline(1.0, color="#A3A095", ls="--", lw=0.8)
    ax2.set_xlabel("σ ruído (Fe, P)")
    ax2.set_ylabel("peso")
    ax2.set_title("memória iônica vs ruído")
    ax2.legend(fontsize=8, frameon=False)
    fig.tight_layout(rect=[0, 0.02, 1, 0.92])
    fig.savefig(outdir / "sim_hifal_ruido.png", dpi=140)
    plt.close(fig)


def sweep_noise(base: Params, outdir: Path) -> list[dict]:
    sigmas = [0.00, 0.04, 0.08, 0.12, 0.16, 0.22, 0.30, 0.40]
    rows = []
    for s in sigmas:
        pp = replace(base, sigma_chem=float(s), steps=min(base.steps, 2600), n_nodes=min(base.n_nodes, 64))
        tissue = HyphalTissue(pp)
        out = tissue.run()
        rows.append(out["noise"])
        print(f"  σ={s:.2f}  true={out['noise']['fire_true']}  "
              f"false={out['noise']['fire_false']}  "
              f"Mmax={out['noise']['max_m']:.2f}  Lmax={out['noise']['max_l']:.2f}")
    plot_noise_sweep(rows, outdir)
    return rows


def main() -> None:
    args = parse_args()
    p = Params(n_nodes=args.nodes, steps=args.steps, dt=args.dt, seed=args.seed,
               sigma_chem=args.sigma)
    tissue = HyphalTissue(p)
    out = tissue.run()
    outdir = Path(args.out)
    plot_results(tissue, out, outdir)
    text = summarize(tissue, out)
    (outdir / "resumo.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\nfiguras: {outdir / 'sim_hifal_arcam.png'}")
    print(f"raster : {outdir / 'sim_hifal_raster.png'}")
    print(f"malha3d: {outdir / 'sim_hifal_3d.png'}")
    print(f"porta  : {outdir / 'sim_hifal_porta.png'}")
    if args.sweep:
        print("\n--- varredura de ruído ---")
        rows = sweep_noise(p, outdir)
        first_false = next((r["sigma"] for r in rows if r["fire_false"] > 0), None)
        print(f"limiar de falso FIRE ≈ σ={first_false}")
        print(f"ruido  : {outdir / 'sim_hifal_ruido.png'}")


if __name__ == "__main__":
    main()
