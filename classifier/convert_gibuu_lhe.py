#!/usr/bin/env python3
"""Convert GiBUU high-energy lepton Les Houches output into HIPO.

The GiBUU high-energy lepton mode writes final-state hadrons after nuclear
transport.  The outgoing electron is reconstructed from the event metadata
(``nu`` and ``Q2``), since the perturbative particle file intentionally stores
the transported hadronic state.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import hipopy
import numpy as np

from generate_rgc_sidis import M_PION, M_PROTON, sample_target_vertex, smear_track


def events(path: Path):
    text = path.read_text()
    for block in re.findall(r"<event>(.*?)</event>", text, flags=re.S):
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        header = lines[0].split()
        if len(header) < 3:
            continue
        particles = []
        metadata = None
        for line in lines[1:]:
            if line.startswith("# 14"):
                fields = line.split()
                if len(fields) >= 6:
                    metadata = [float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5]), int(fields[6])]
                continue
            fields = line.split()
            if len(fields) < 10:
                continue
            particles.append({"pid": int(fields[0]), "px": float(fields[6]), "py": float(fields[7]),
                              "pz": float(fields[8]), "energy": float(fields[9])})
        if metadata is not None:
            yield float(header[2]), metadata, particles


def write_sample(source: Path, output_path: Path, target: str, seed: int, limit: int | None) -> int:
    rng = np.random.default_rng(seed)
    output = hipopy.create(str(output_path))
    output.newTree("SIM::Event", {
        "event": "I", "target": "B", "beam_helicity": "B", "target_spin": "B", "npart": "S",
        "x": "F", "q2": "F", "w": "F", "y": "F", "z": "F", "phi_h": "F", "p_t": "F",
        "a_ll": "F", "weight": "F", "vx": "F", "vy": "F", "vz": "F"}, group=1000, item=1)
    output.newTree("MC::Particle", {"pid": "I", "px": "F", "py": "F", "pz": "F", "charge": "B"}, group=1000, item=2)
    output.newTree("REC::Particle", {"pid": "I", "px": "F", "py": "F", "pz": "F", "charge": "B", "status": "S", "vx": "F", "vy": "F", "vz": "F"}, group=300, item=1)
    output.newTree("REC::Event", {"event": "I", "x": "F", "q2": "F", "w": "F", "y": "F", "z": "F", "phi_h": "F", "p_t": "F", "mm2_e_pi": "F", "mm_e_pi": "F", "vz": "F"}, group=300, item=2)
    output.open()

    beam_energy = 10.6
    beam = np.array([0.0, 0.0, beam_energy])
    target_code = {"hydrogen": 1, "nitrogen": 2, "carbon": 3, "deuterium": 4}[target]
    accepted = 0
    try:
        for weight, metadata, particles in events(source):
            if limit is not None and accepted >= limit:
                break
            nu, q2, _eps, phi_l, _event_type = metadata
            eprime = beam_energy - nu
            if eprime <= 0.2 or q2 <= 0:
                continue
            cos_theta = 1.0 - q2 / (2.0 * beam_energy * eprime)
            if not -1.0 < cos_theta < 1.0:
                continue
            theta = np.arccos(cos_theta)
            e_truth = eprime * np.array([np.sin(theta) * np.cos(phi_l), np.sin(theta) * np.sin(phi_l), np.cos(theta)])
            pion = next((p for p in particles if p["pid"] in (211, -211) and p["energy"] > M_PION), None)
            if pion is None:
                continue
            pi_truth = np.array([pion["px"], pion["py"], pion["pz"]])
            q = beam - e_truth
            qhat = q / np.linalg.norm(q)
            pi_energy = np.sqrt(np.dot(pi_truth, pi_truth) + M_PION**2)
            x = q2 / (2.0 * M_PROTON * nu)
            w2 = M_PROTON**2 + 2.0 * M_PROTON * nu - q2
            if w2 <= 0:
                continue
            z = pi_energy / nu
            p_t = np.linalg.norm(pi_truth - np.dot(pi_truth, qhat) * qhat)
            yhat = np.cross(np.array([0.0, 0.0, 1.0]), qhat)
            if np.linalg.norm(yhat) < 1e-8:
                continue
            yhat /= np.linalg.norm(yhat)
            xhat = np.cross(yhat, qhat)
            phi_h = np.arctan2(np.dot(pi_truth, yhat), np.dot(pi_truth, xhat))
            e_reco, pi_reco = smear_track(e_truth, rng), smear_track(pi_truth, rng)
            vertex_truth = sample_target_vertex(rng)
            vertex_reco = vertex_truth + rng.normal(0.0, [0.05, 0.05, 0.20])

            e_rec = np.linalg.norm(e_reco)
            q_rec = beam - e_reco
            q2_rec = 2.0 * beam_energy * e_rec * (1.0 - e_reco[2] / e_rec)
            nu_rec = beam_energy - e_rec
            if nu_rec <= 0:
                continue
            qhat_rec = q_rec / np.linalg.norm(q_rec)
            x_rec = q2_rec / (2.0 * M_PROTON * nu_rec)
            w2_rec = M_PROTON**2 + 2.0 * M_PROTON * nu_rec - q2_rec
            pi_e_rec = np.sqrt(np.dot(pi_reco, pi_reco) + M_PION**2)
            z_rec = pi_e_rec / nu_rec
            pt_rec = np.linalg.norm(pi_reco - np.dot(pi_reco, qhat_rec) * qhat_rec)
            yhat_rec = np.cross(np.array([0.0, 0.0, 1.0]), qhat_rec)
            yhat_rec /= np.linalg.norm(yhat_rec)
            xhat_rec = np.cross(yhat_rec, qhat_rec)
            phi_rec = np.arctan2(np.dot(pi_reco, yhat_rec), np.dot(pi_reco, xhat_rec))
            initial_p = rng.normal(0.0, 0.15, 3)
            initial_e = np.sqrt(M_PROTON**2 + np.dot(initial_p, initial_p))
            miss_e = beam_energy + initial_e - e_rec - pi_e_rec
            miss_p = beam + initial_p - e_reco - pi_reco
            mm2 = miss_e**2 - np.dot(miss_p, miss_p)
            pids = np.array([11, pion["pid"]])

            output.writeBank("SIM::Event", ["event", "target", "beam_helicity", "target_spin", "npart", "x", "q2", "w", "y", "z", "phi_h", "p_t", "a_ll", "weight", "vx", "vy", "vz"],
                [np.array([accepted]), np.array([target_code]), np.array([rng.choice([-1, 1])]), np.array([0]), np.array([2]), np.array([x]), np.array([q2]), np.array([np.sqrt(w2)]), np.array([nu / beam_energy]), np.array([z]), np.array([phi_h]), np.array([p_t]), np.array([0.0]), np.array([weight]), np.array([vertex_truth[0]]), np.array([vertex_truth[1]]), np.array([vertex_truth[2]])], dtypes="IBBBSFFFFFFFFFFFFF")
            charges = np.array([-1, 1 if pion["pid"] == 211 else -1])
            output.writeBank("MC::Particle", ["pid", "px", "py", "pz", "charge"], [pids, np.array([e_truth[0], pi_truth[0]]), np.array([e_truth[1], pi_truth[1]]), np.array([e_truth[2], pi_truth[2]]), charges], dtypes="IFFFB")
            output.writeBank("REC::Particle", ["pid", "px", "py", "pz", "charge", "status", "vx", "vy", "vz"], [pids, np.array([e_reco[0], pi_reco[0]]), np.array([e_reco[1], pi_reco[1]]), np.array([e_reco[2], pi_reco[2]]), charges, np.array([200, 200]), np.array([vertex_reco[0], vertex_reco[0]]), np.array([vertex_reco[1], vertex_reco[1]]), np.array([vertex_reco[2], vertex_reco[2]])], dtypes="IFFFBSFFF")
            output.writeBank("REC::Event", ["event", "x", "q2", "w", "y", "z", "phi_h", "p_t", "mm2_e_pi", "mm_e_pi", "vz"], [np.array([accepted]), np.array([x_rec]), np.array([q2_rec]), np.array([np.sqrt(max(w2_rec, 0.0))]), np.array([nu_rec / beam_energy]), np.array([z_rec]), np.array([phi_rec]), np.array([pt_rec]), np.array([mm2]), np.array([np.sqrt(max(mm2, 0.0))]), np.array([vertex_reco[2]])], dtypes="IFFFFFFFFFF")
            output.addEvent()
            output.event.reset()
            accepted += 1
    finally:
        output.close()
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target", choices=("hydrogen", "nitrogen", "carbon", "deuterium"), required=True)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    count = write_sample(args.source, args.output, args.target, args.seed, args.limit)
    print(f"Wrote {count} GiBUU-derived {args.target} events to {args.output}")


if __name__ == "__main__":
    main()
