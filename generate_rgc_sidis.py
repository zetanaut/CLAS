#!/usr/bin/env python3
"""Generate a small Run Group C-inspired longitudinal SIDIS toy sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import hipopy
import numpy as np


M_PROTON = 0.938272
M_PION = 0.13957


def asymmetry_model(x: float, z: float, q2: float, pid: int) -> float:
    """Toy A_LL model; this is intentionally not a physics fit."""
    charge_factor = 1.0 if pid == 211 else 0.8
    return charge_factor * 0.10 * (0.35 + x) * np.exp(-0.04 * (q2 - 1.0))


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def in_forward_sector(phi: float, gap_degrees: float = 3.0) -> bool:
    """Approximate six-sector acceptance; gap_degrees is a toy fiducial cut."""
    sector_local = ((np.degrees(phi) + 30.0) % 60.0) - 30.0
    return abs(sector_local) < (30.0 - gap_degrees)


def smear_track(vector: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Smear a forward track with CLAS12-like p, theta, and phi resolutions."""
    momentum = np.linalg.norm(vector)
    theta = np.arccos(vector[2] / momentum)
    phi = np.arctan2(vector[1], vector[0])
    # Forward tracking target: approximately 1% at 5 GeV.
    sigma_p_over_p = np.sqrt(0.005**2 + (0.001 * momentum) ** 2)
    sigma_theta = 0.001  # rad, approximately 1 mrad
    sigma_phi = 0.004    # rad, approximately 4 mrad
    p_reco = momentum * (1.0 + rng.normal(0.0, sigma_p_over_p))
    theta_reco = theta + rng.normal(0.0, sigma_theta)
    phi_reco = phi + rng.normal(0.0, sigma_phi)
    return p_reco * np.array([
        np.sin(theta_reco) * np.cos(phi_reco),
        np.sin(theta_reco) * np.sin(phi_reco),
        np.cos(theta_reco),
    ])


def sample_target_vertex(rng: np.random.Generator) -> np.ndarray:
    """Sample the compact RGC target/raster volume in centimetres."""
    radius = 0.95 * np.sqrt(rng.random())  # 1.9 cm diameter SIDIS raster
    azimuth = rng.uniform(-np.pi, np.pi)
    return np.array([
        radius * np.cos(azimuth),
        radius * np.sin(azimuth),
        rng.uniform(-2.5, 2.5),  # approximately 5 cm target-cell length
    ])


def generate(path: Path, events: int, seed: int = 12345, beam_energy: float = 10.6, material: str = "nh3") -> None:
    """Write accepted simulated SIDIS events to *path*."""
    if material not in {"nh3", "carbon"}:
        raise ValueError("material must be 'nh3' or 'carbon'")
    rng = np.random.default_rng(seed)
    output = hipopy.create(str(path))
    output.newTree(
        "SIM::Event",
        {
            "event": "I", "target": "B", "beam_helicity": "B", "target_spin": "B",
            "npart": "S", "x": "F", "q2": "F", "w": "F", "y": "F",
            "z": "F", "phi_h": "F", "p_t": "F", "a_ll": "F",
            "vx": "F", "vy": "F", "vz": "F",
        },
        group=1000, item=1,
    )
    output.newTree(
        "MC::Particle",
        {"pid": "I", "px": "F", "py": "F", "pz": "F", "charge": "B"},
        group=1000, item=2,
    )
    output.newTree(
        "REC::Particle",
        {"pid": "I", "px": "F", "py": "F", "pz": "F", "charge": "B", "status": "S", "vx": "F", "vy": "F", "vz": "F"},
        group=300, item=1,
    )
    output.newTree(
        "REC::Event",
        {"event": "I", "x": "F", "q2": "F", "w": "F", "y": "F", "z": "F", "phi_h": "F", "p_t": "F", "mm2_e_pi": "F", "mm_e_pi": "F", "vz": "F"},
        group=300, item=2,
    )
    output.open()

    beam_polarization = 0.85
    target_polarization = 0.80
    beam = np.array([0.0, 0.0, beam_energy])
    beam_hat = unit(beam)
    accepted = 0
    trials = 0

    try:
        while accepted < events:
            trials += 1
            x = float(rng.uniform(0.10, 0.60))
            y = float(rng.uniform(0.25, 0.75))
            q2 = 2.0 * M_PROTON * beam_energy * x * y
            nu = beam_energy * y
            w2 = M_PROTON**2 + 2.0 * M_PROTON * nu - q2
            if q2 <= 1.0 or w2 <= 4.0:
                continue

            scattered_energy = beam_energy * (1.0 - y)
            sin_half = np.sqrt(q2 / (4.0 * beam_energy * scattered_energy))
            if sin_half >= 1.0:
                continue
            theta_e = 2.0 * np.arcsin(sin_half)
            if not np.deg2rad(5.0) < theta_e < np.deg2rad(35.0):
                continue

            electron_truth = np.array([
                scattered_energy * np.sin(theta_e), 0.0,
                scattered_energy * np.cos(theta_e),
            ])
            q = beam - electron_truth
            q_hat = unit(q)
            y_hat = unit(np.cross(beam_hat, q_hat))
            x_hat = unit(np.cross(y_hat, q_hat))

            pid = int(rng.choice([211, -211]))
            z = float(rng.uniform(0.30, 0.70))
            phi_h = float(rng.uniform(-np.pi, np.pi))
            # NH3 uses an effective 3/17 free-proton fraction. Carbon is a
            # pure unpolarized nuclear control sample.
            if material == "nh3":
                target = 1 if rng.random() < (3.0 / 17.0) else 2  # 1=H, 2=N
            else:
                target = 3  # C
            p_t_scale = {1: 0.32, 2: 0.45, 3: 0.48}[target]
            p_t = float(rng.rayleigh(p_t_scale))
            pion_energy = z * nu
            if pion_energy <= M_PION or p_t >= np.sqrt(pion_energy**2 - M_PION**2):
                continue
            p_long = np.sqrt(pion_energy**2 - M_PION**2 - p_t**2)
            pion_truth = (
                p_long * q_hat
                + p_t * np.cos(phi_h) * x_hat
                + p_t * np.sin(phi_h) * y_hat
            )

            pion_theta = np.arccos(pion_truth[2] / np.linalg.norm(pion_truth))
            if not np.deg2rad(5.0) < pion_theta < np.deg2rad(40.0):
                continue
            if np.linalg.norm(pion_truth) < 0.20:
                continue
            if not in_forward_sector(np.arctan2(electron_truth[1], electron_truth[0])):
                continue
            if not in_forward_sector(np.arctan2(pion_truth[1], pion_truth[0])):
                continue

            target_spin = int(rng.choice([-1, 1]))
            beam_helicity = int(rng.choice([-1, 1]))
            # Only the free protons in NH3 carry the target polarization.
            a_ll = asymmetry_model(x, z, q2, pid) if target == 1 else 0.0
            probability = 0.5 * (
                1.0 + beam_helicity * target_spin * beam_polarization * target_polarization * a_ll
            )
            if rng.random() > probability / 0.5:
                continue
            if rng.random() > 0.98 or rng.random() > 0.94:
                continue  # electron and pion tracking/PID efficiency toy

            electron_reco = smear_track(electron_truth, rng)
            pion_reco = smear_track(pion_truth, rng)
            charge = 1 if pid == 211 else -1
            vertex_truth = sample_target_vertex(rng)
            vertex_reco = vertex_truth + rng.normal(0.0, [0.05, 0.05, 0.20])

            # Recompute reconstructed DIS/SIDIS variables from the smeared tracks.
            reco_energy = np.linalg.norm(electron_reco)
            reco_q = beam - electron_reco
            reco_q2 = 2.0 * beam_energy * reco_energy * (1.0 - np.dot(beam_hat, unit(electron_reco)))
            reco_nu = beam_energy - reco_energy
            reco_y = reco_nu / beam_energy
            reco_x = reco_q2 / (2.0 * M_PROTON * reco_nu)
            reco_w2 = M_PROTON**2 + 2.0 * M_PROTON * reco_nu - reco_q2
            reco_qhat = unit(reco_q)
            reco_pion_energy = np.sqrt(np.dot(pion_reco, pion_reco) + M_PION**2)
            reco_z = reco_pion_energy / reco_nu
            reco_pt = np.linalg.norm(pion_reco - np.dot(pion_reco, reco_qhat) * reco_qhat)
            reco_yhat = unit(np.cross(beam_hat, reco_qhat))
            reco_xhat = unit(np.cross(reco_yhat, reco_qhat))
            reco_phi = np.arctan2(np.dot(pion_reco, reco_yhat), np.dot(pion_reco, reco_xhat))
            # Missing mass of the detected e' + pi system. For this SIDIS
            # sample it represents X and should be broad, not proton-peaked.
            initial_p = np.zeros(3) if target == 1 else rng.normal(0.0, 0.15, 3)
            initial_energy = np.sqrt(M_PROTON**2 + np.dot(initial_p, initial_p))
            missing_energy = beam_energy + initial_energy - reco_energy - reco_pion_energy
            missing_momentum = beam + initial_p - electron_reco - pion_reco
            missing_mass2 = missing_energy**2 - np.dot(missing_momentum, missing_momentum)
            missing_mass = np.sqrt(max(missing_mass2, 0.0))

            output.writeBank(
                "SIM::Event",
                ["event", "target", "beam_helicity", "target_spin", "npart", "x", "q2", "w", "y", "z", "phi_h", "p_t", "a_ll"],
                [
                    np.array([accepted]), np.array([target]), np.array([beam_helicity]),
                    np.array([target_spin]), np.array([2]), np.array([x]), np.array([q2]),
                    np.array([np.sqrt(w2)]), np.array([y]), np.array([z]), np.array([phi_h]),
                    np.array([p_t]), np.array([a_ll]),
                ],
                dtypes="IBBBSFFFFFFFFFFF",
            )
            output.writeBank(
                "MC::Particle",
                ["pid", "px", "py", "pz", "charge"],
                [np.array([11, pid]), np.array([electron_truth[0], pion_truth[0]]),
                 np.array([electron_truth[1], pion_truth[1]]), np.array([electron_truth[2], pion_truth[2]]),
                 np.array([-1, charge])],
                dtypes="IFFFB",
            )
            output.writeBank(
                "REC::Particle",
                ["pid", "px", "py", "pz", "charge", "status", "vx", "vy", "vz"],
                [np.array([11, pid]), np.array([electron_reco[0], pion_reco[0]]),
                 np.array([electron_reco[1], pion_reco[1]]), np.array([electron_reco[2], pion_reco[2]]),
                 np.array([-1, charge]), np.array([200, 200]),
                 np.array([vertex_reco[0], vertex_reco[0]]), np.array([vertex_reco[1], vertex_reco[1]]),
                 np.array([vertex_reco[2], vertex_reco[2]])],
                dtypes="IFFFBSFFF",
            )
            output.writeBank(
                "REC::Event",
                ["event", "x", "q2", "w", "y", "z", "phi_h", "p_t", "mm2_e_pi", "mm_e_pi", "vz"],
                [np.array([accepted]), np.array([reco_x]), np.array([reco_q2]), np.array([np.sqrt(max(reco_w2, 0.0))]),
                 np.array([reco_y]), np.array([reco_z]), np.array([reco_phi]), np.array([reco_pt]),
                 np.array([missing_mass2]), np.array([missing_mass]), np.array([vertex_reco[2]])],
                dtypes="IFFFFFFFFFF",
            )
            output.addEvent()
            output.event.reset()
            accepted += 1
    finally:
        output.close()

    print(f"Wrote {accepted} accepted {material} SIDIS events to {path} ({trials} trials)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("rgc_sidis.hipo"))
    parser.add_argument("--events", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--beam-energy", type=float, default=10.6)
    parser.add_argument("--target", choices=("nh3", "carbon", "both"), default="nh3")
    args = parser.parse_args()
    if args.target == "both":
        generate(args.output.with_name(args.output.stem + "_nh3" + args.output.suffix), args.events, args.seed, args.beam_energy, "nh3")
        generate(args.output.with_name(args.output.stem + "_carbon" + args.output.suffix), args.events, args.seed + 1, args.beam_energy, "carbon")
    else:
        generate(args.output, args.events, args.seed, args.beam_energy, args.target)


if __name__ == "__main__":
    main()
