#!/usr/bin/env python3
"""Convert a clasDIS LUND file into a compact, detector-smeared HIPO sample.

clasDIS supplies the physics event generation and hadronization.  This bridge
keeps the generated electron and charged pion, applies the same lightweight
CLAS12 resolution model as the Python study, and labels NH3/C events for the
classifier exercise.  Nitrogen and carbon are effective nuclear-background
labels here; this is not a nuclear transport calculation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hipopy
import numpy as np

from generate_rgc_sidis import (
    M_PION,
    M_PROTON,
    asymmetry_model,
    sample_target_vertex,
    smear_track,
)


def parse_lund(path: Path):
    """Yield ``(header, particle_rows)`` from a clasDIS LUND text file."""
    lines = path.read_text().splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].strip():
            index += 1
            continue
        header = lines[index].split()
        index += 1
        if len(header) < 10:
            continue
        npart = int(header[0])
        rows = []
        for _ in range(npart):
            if index >= len(lines):
                break
            fields = lines[index].split()
            index += 1
            if len(fields) >= 13:
                rows.append(fields)
        if len(rows) == npart:
            yield header, rows


def particle(row):
    return {
        "index": int(row[0]),
        "charge": int(float(row[1])),
        "status": int(row[2]),
        "pid": int(row[3]),
        "px": float(row[6]),
        "py": float(row[7]),
        "pz": float(row[8]),
        "energy": float(row[9]),
    }


def source_kind(path: Path) -> str:
    """Infer the elementary target from clasDIS's standard filename."""
    return "neutron" if "clasdisne" in path.name else "proton"


def write_sample(output_path: Path, lund_paths: list[Path], material: str, seed: int, limit: int | None) -> int:
    rng = np.random.default_rng(seed)
    records = []
    for lund_path in lund_paths:
        records.extend((header, rows, source_kind(lund_path)) for header, rows in parse_lund(lund_path))
    if not records:
        raise ValueError("no complete LUND events found")
    by_source = {
        "proton": [record for record in records if record[2] == "proton"],
        "neutron": [record for record in records if record[2] == "neutron"],
    }
    if not by_source["proton"] and not by_source["neutron"]:
        raise ValueError("could not identify proton/neutron LUND inputs")
    output = hipopy.create(str(output_path))
    output.newTree(
        "SIM::Event",
        {"event": "I", "target": "B", "beam_helicity": "B", "target_spin": "B",
         "npart": "S", "x": "F", "q2": "F", "w": "F", "y": "F", "z": "F",
         "phi_h": "F", "p_t": "F", "a_ll": "F", "weight": "F", "vx": "F",
         "vy": "F", "vz": "F"},
        group=1000, item=1,
    )
    output.newTree(
        "MC::Particle",
        {"pid": "I", "px": "F", "py": "F", "pz": "F", "charge": "B"},
        group=1000, item=2,
    )
    output.newTree(
        "REC::Particle",
        {"pid": "I", "px": "F", "py": "F", "pz": "F", "charge": "B", "status": "S",
         "vx": "F", "vy": "F", "vz": "F"},
        group=300, item=1,
    )
    output.newTree(
        "REC::Event",
        {"event": "I", "x": "F", "q2": "F", "w": "F", "y": "F", "z": "F",
         "phi_h": "F", "p_t": "F", "mm2_e_pi": "F", "mm_e_pi": "F", "vz": "F"},
        group=300, item=2,
    )
    output.open()

    beam_energy = 10.6
    beam = np.array([0.0, 0.0, beam_energy])
    accepted = 0
    try:
        target_events = limit if limit is not None else len(records)
        for _ in range(target_events):
            if material == "proton":
                target, source = 1, "proton"
            elif material == "neutron":
                target, source = 2, "neutron"
            elif material == "nitrogen":
                target, source = 2, str(rng.choice(["proton", "neutron"]))
            elif material == "carbon":
                target, source = 3, str(rng.choice(["proton", "neutron"]))
            else:  # NH3: three polarized free protons plus a 14-nucleon N background.
                target = 1 if rng.random() < 3.0 / 17.0 else 2
                source = "proton" if target == 1 else str(rng.choice(["proton", "neutron"]))
            pool = by_source[source]
            if not pool:
                raise ValueError(f"missing {source} LUND input for material={material}")
            header, rows, source = pool[int(rng.integers(0, len(pool)))]
            parts = [particle(row) for row in rows]
            electron = next((p for p in parts if p["pid"] == 11 and p["charge"] < 0 and p["status"] == 1), None)
            pion = next((p for p in parts if p["pid"] in (211, -211) and p["status"] == 1), None)
            if electron is None or pion is None:
                continue

            e_truth = np.array([electron["px"], electron["py"], electron["pz"]])
            pi_truth = np.array([pion["px"], pion["py"], pion["pz"]])
            e_reco = smear_track(e_truth, rng)
            pi_reco = smear_track(pi_truth, rng)
            beam_helicity = int(rng.choice([-1, 1]))
            target_spin = int(rng.choice([-1, 1]))

            e = np.linalg.norm(e_truth)
            q = beam - e_truth
            q2 = 2.0 * beam_energy * e * (1.0 - e_truth[2] / e)
            nu = beam_energy - e
            if nu <= 0 or q2 <= 0:
                continue
            x = q2 / (2.0 * M_PROTON * nu)
            w2 = M_PROTON**2 + 2.0 * M_PROTON * nu - q2
            if w2 <= 0:
                continue
            qhat = q / np.linalg.norm(q)
            pi_energy = np.sqrt(np.dot(pi_truth, pi_truth) + M_PION**2)
            z = pi_energy / nu
            p_t = np.linalg.norm(pi_truth - np.dot(pi_truth, qhat) * qhat)
            yhat = np.cross(np.array([0.0, 0.0, 1.0]), qhat)
            if np.linalg.norm(yhat) < 1e-8:
                continue
            yhat /= np.linalg.norm(yhat)
            xhat = np.cross(yhat, qhat)
            phi_h = np.arctan2(np.dot(pi_truth, yhat), np.dot(pi_truth, xhat))
            a_ll = asymmetry_model(x, z, q2, pion["pid"]) if target == 1 else 0.0
            vertex_truth = sample_target_vertex(rng)
            vertex_reco = vertex_truth + rng.normal(0.0, [0.05, 0.05, 0.20])

            e_rec = np.linalg.norm(e_reco)
            q_rec = beam - e_reco
            q2_rec = 2.0 * beam_energy * e_rec * (1.0 - e_reco[2] / e_rec)
            nu_rec = beam_energy - e_rec
            if nu_rec <= 0:
                continue
            x_rec = q2_rec / (2.0 * M_PROTON * nu_rec)
            w2_rec = M_PROTON**2 + 2.0 * M_PROTON * nu_rec - q2_rec
            qhat_rec = q_rec / np.linalg.norm(q_rec)
            pi_e_rec = np.sqrt(np.dot(pi_reco, pi_reco) + M_PION**2)
            z_rec = pi_e_rec / nu_rec
            pt_rec = np.linalg.norm(pi_reco - np.dot(pi_reco, qhat_rec) * qhat_rec)
            yhat_rec = np.cross(np.array([0.0, 0.0, 1.0]), qhat_rec)
            yhat_rec /= np.linalg.norm(yhat_rec)
            xhat_rec = np.cross(yhat_rec, qhat_rec)
            phi_rec = np.arctan2(np.dot(pi_reco, yhat_rec), np.dot(pi_reco, xhat_rec))
            initial_p = np.zeros(3) if target == 1 else rng.normal(0.0, 0.15, 3)
            initial_e = np.sqrt(M_PROTON**2 + np.dot(initial_p, initial_p))
            miss_e = beam_energy + initial_e - e_rec - pi_e_rec
            miss_p = beam + initial_p - e_reco - pi_reco
            mm2 = miss_e**2 - np.dot(miss_p, miss_p)
            weight = float(header[9])

            output.writeBank("SIM::Event",
                ["event", "target", "beam_helicity", "target_spin", "npart", "x", "q2", "w", "y", "z", "phi_h", "p_t", "a_ll", "weight", "vx", "vy", "vz"],
                [np.array([accepted]), np.array([target]), np.array([beam_helicity]), np.array([target_spin]), np.array([2]), np.array([x]), np.array([q2]), np.array([np.sqrt(w2)]), np.array([nu / beam_energy]), np.array([z]), np.array([phi_h]), np.array([p_t]), np.array([a_ll]), np.array([weight]), np.array([vertex_truth[0]]), np.array([vertex_truth[1]]), np.array([vertex_truth[2]])],
                dtypes="IBBBSFFFFFFFFFFFFF")
            pids = np.array([11, pion["pid"]])
            output.writeBank("MC::Particle", ["pid", "px", "py", "pz", "charge"],
                [pids, np.array([e_truth[0], pi_truth[0]]), np.array([e_truth[1], pi_truth[1]]), np.array([e_truth[2], pi_truth[2]]), np.array([-1, 1 if pion["pid"] == 211 else -1])], dtypes="IFFFB")
            output.writeBank("REC::Particle", ["pid", "px", "py", "pz", "charge", "status", "vx", "vy", "vz"],
                [pids, np.array([e_reco[0], pi_reco[0]]), np.array([e_reco[1], pi_reco[1]]), np.array([e_reco[2], pi_reco[2]]), np.array([-1, 1 if pion["pid"] == 211 else -1]), np.array([200, 200]), np.array([vertex_reco[0], vertex_reco[0]]), np.array([vertex_reco[1], vertex_reco[1]]), np.array([vertex_reco[2], vertex_reco[2]])], dtypes="IFFFBSFFF")
            output.writeBank("REC::Event", ["event", "x", "q2", "w", "y", "z", "phi_h", "p_t", "mm2_e_pi", "mm_e_pi", "vz"],
                [np.array([accepted]), np.array([x_rec]), np.array([q2_rec]), np.array([np.sqrt(max(w2_rec, 0.0))]), np.array([nu_rec / beam_energy]), np.array([z_rec]), np.array([phi_rec]), np.array([pt_rec]), np.array([mm2]), np.array([np.sqrt(max(mm2, 0.0))]), np.array([vertex_reco[2]])], dtypes="IFFFFFFFFFF")
            output.addEvent()
            output.event.reset()
            accepted += 1
    finally:
        output.close()
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lund", type=Path, nargs="+", help="one or more clasDIS LUND files")
    parser.add_argument("output", type=Path)
    parser.add_argument("--material", choices=("nh3", "nitrogen", "carbon", "proton", "neutron"), default="nh3")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    count = write_sample(args.output, args.lund, args.material, args.seed, args.limit)
    print(f"Wrote {count} clasDIS-derived {args.material} events to {args.output}")


if __name__ == "__main__":
    main()
