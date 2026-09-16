#!/usr/bin/env python3
"""Make cross-section-weighted, unnormalized D/N yield comparisons."""

from __future__ import annotations

import argparse
from pathlib import Path

import hipopy
import matplotlib.pyplot as plt
import numpy as np


def read_events(path: str) -> dict[str, np.ndarray]:
    names = ("x", "q2", "w", "y", "z", "p_t", "phi_h", "pion_p",
             "pion_theta", "mm2_e_pi", "mm_e_pi", "vz", "rxy", "weight")
    out = {name: [] for name in names}
    f = hipopy.open(path)
    f.readBank("SIM::Event")
    f.readBank("REC::Particle")
    f.readBank("REC::Event")
    try:
        while f.nextEvent():
            f.event.read(f.banklist["SIM::Event"])
            f.event.read(f.banklist["REC::Particle"])
            f.event.read(f.banklist["REC::Event"])
            px, py, pz = (f.getFloats("REC::Particle", k) for k in ("px", "py", "pz"))
            p = float(np.sqrt(px[1] ** 2 + py[1] ** 2 + pz[1] ** 2))
            theta = float(np.degrees(np.arccos(pz[1] / p)))
            vx, vy = f.getFloats("REC::Particle", "vx")[:1], f.getFloats("REC::Particle", "vy")[:1]
            for name in ("x", "q2", "w", "y", "z", "phi_h", "p_t", "mm2_e_pi", "mm_e_pi", "vz"):
                out[name].append(float(f.getFloats("REC::Event", name)[0]))
            out["pion_p"].append(p)
            out["pion_theta"].append(theta)
            out["rxy"].append(float(np.sqrt(vx[0] ** 2 + vy[0] ** 2)))
            out["weight"].append(float(f.getFloats("SIM::Event", "weight")[0]))
    finally:
        f.close()
    return {key: np.asarray(value) for key, value in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deuterium")
    parser.add_argument("nitrogen")
    parser.add_argument("--output", type=Path, default=Path("rgc_gibuu_nd3_yields.png"))
    args = parser.parse_args()
    d, n = read_events(args.deuterium), read_events(args.nitrogen)
    specs = [
        ("x", "x", (0, .65)), ("q2", r"$Q^2$ [GeV$^2$]", (0, 10)),
        ("w", r"$W$ [GeV]", (1.5, 5)), ("y", "y", (0, 1)),
        ("z", "z", (0, 1)), ("p_t", r"$p_T$ [GeV]", (0, 1.5)),
        ("phi_h", r"$\phi_h$ [rad]", (-np.pi, np.pi)),
        ("pion_p", r"$p_\pi$ [GeV]", (0, 5)),
        ("pion_theta", r"$\theta_\pi$ [deg]", (0, 45)),
        ("mm2_e_pi", r"$M_X^2(e'\pi)$ [GeV$^2$]", (-2, 12)),
        ("mm_e_pi", r"$M_X(e'\pi)$ [GeV]", (0, 4)),
        ("vz", r"$v_z$ [cm]", (-3.5, 3.5)),
    ]
    fig, axes = plt.subplots(4, 3, figsize=(14, 16), constrained_layout=True)
    # ND3 has three deuterons per nitrogen nucleus. GiBUU event weights carry
    # the generated cross-section weight; retain absolute weighted yields.
    for axis, (name, label, limits) in zip(axes.flat, specs):
        axis.hist(d[name], bins=30, range=limits, weights=3.0 * d["weight"],
                  histtype="step", label="3D", color="tab:red")
        axis.hist(n[name], bins=30, range=limits, weights=n["weight"],
                  histtype="step", label="N", color="tab:orange")
        axis.set_xlabel(label)
        axis.set_ylabel("weighted yield")
        axis.legend(fontsize=8)
    fig.suptitle("GiBUU ND$_3$-like cross-section-weighted yields")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")
    print(f"Weighted yields: 3D={3*d['weight'].sum():.3g}, N={n['weight'].sum():.3g}")


if __name__ == "__main__":
    main()
