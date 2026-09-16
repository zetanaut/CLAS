#!/usr/bin/env python3
"""Make common kinematic and spin plots from the SIDIS HIPO toy sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import hipopy
import matplotlib.pyplot as plt
import numpy as np


def read_events(path: str) -> dict[str, np.ndarray]:
    file = hipopy.open(path)
    names = ("target", "product", "x", "q2", "w", "y", "z", "phi_h", "p_t", "pion_p", "pion_theta", "mm2_e_pi", "mm_e_pi", "vz", "rxy")
    columns = {name: [] for name in names}
    file.readBank("SIM::Event")
    file.readBank("REC::Particle")
    file.readBank("REC::Event")
    try:
        while file.nextEvent():
            file.event.read(file.banklist["SIM::Event"])
            file.event.read(file.banklist["REC::Particle"])
            file.event.read(file.banklist["REC::Event"])
            target = int(file.getBytes("SIM::Event", "target")[0])
            helicity = int(file.getBytes("SIM::Event", "beam_helicity")[0])
            spin = int(file.getBytes("SIM::Event", "target_spin")[0])
            px = file.getFloats("REC::Particle", "px")
            py = file.getFloats("REC::Particle", "py")
            pz = file.getFloats("REC::Particle", "pz")
            vx = file.getFloats("REC::Particle", "vx")
            vy = file.getFloats("REC::Particle", "vy")
            pion_p = float(np.sqrt(px[1] ** 2 + py[1] ** 2 + pz[1] ** 2))
            pion_theta = float(np.degrees(np.arccos(pz[1] / pion_p)))
            columns["target"].append(target)
            columns["product"].append(helicity * spin)
            for name in ("x", "q2", "w", "y", "z", "phi_h", "p_t"):
                columns[name].append(file.getFloats("REC::Event", name)[0])
            columns["pion_p"].append(pion_p)
            columns["pion_theta"].append(pion_theta)
            columns["mm2_e_pi"].append(file.getFloats("REC::Event", "mm2_e_pi")[0])
            columns["mm_e_pi"].append(file.getFloats("REC::Event", "mm_e_pi")[0])
            columns["vz"].append(file.getFloats("REC::Event", "vz")[0])
            columns["rxy"].append(float(np.sqrt(vx[0] ** 2 + vy[0] ** 2)))
    finally:
        file.close()
    return {key: np.asarray(value) for key, value in columns.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", type=Path, default=Path("rgc_sidis_distributions.png"))
    args = parser.parse_args()
    data = read_events(args.input)

    specs = [
        ("x", "x", (0, 0.65)), ("q2", r"$Q^2$ [GeV$^2$]", (0, 10)),
        ("w", r"$W$ [GeV]", (1.5, 5.0)), ("y", "y", (0, 1)),
        ("z", "z", (0, 1)), ("p_t", r"$p_T$ [GeV]", (0, 1.5)),
        ("phi_h", r"$\phi_h$ [rad]", (-np.pi, np.pi)),
        ("pion_p", r"$p_{\pi}$ [GeV]", (0, 5)),
        ("pion_theta", r"$\theta_{\pi}$ [deg]", (0, 45)),
        ("mm2_e_pi", r"$M_X^2(e'\pi)$ [GeV$^2$]", (-2, 12)),
        ("mm_e_pi", r"$M_X(e'\pi)$ [GeV]", (0, 4)),
        ("vz", r"$v_z$ [cm]", (-3.5, 3.5)),
        ("rxy", r"$r_{xy}$ [cm]", (0, 1.5)),
    ]
    fig, axes = plt.subplots(5, 3, figsize=(14, 18), constrained_layout=True)
    masks = [("all", np.ones(len(data["target"]), dtype=bool), "black")]
    for legend, target, color in (("H", 1, "tab:blue"), ("N", 2, "tab:orange"), ("C", 3, "tab:green"), ("D", 4, "tab:red")):
        mask = data["target"] == target
        if np.any(mask):
            masks.append((legend, mask, color))
    for axis, (name, label, limits) in zip(axes.flat, specs):
        for legend, mask, color in masks:
            axis.hist(data[name][mask], bins=30, range=limits, histtype="step", density=True, label=legend, color=color)
        axis.set_xlabel(label)
        axis.set_ylabel("normalized counts")
        axis.legend(fontsize=8)
    for axis in axes.flat[len(specs):]:
        axis.set_visible(False)
    title = "Run Group C-inspired SIDIS toy"
    if np.any(data["target"] == 2) and np.any(data["target"] == 4):
        title += " — ND$_3$-like D/N"
    if np.any(data["target"] == 1):
        title += " — polarized NH$_3$"
    if np.any(data["target"] == 3):
        title += " — carbon control"
    fig.suptitle(title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
