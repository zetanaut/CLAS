#!/usr/bin/env python3
"""Measure the toy longitudinal SIDIS asymmetry from a HIPO file."""

from __future__ import annotations

import argparse

import hipopy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()
    file = hipopy.open(args.input)
    file.readBank("SIM::Event")
    file.readBank("REC::Particle")

    counts = {
        "all": {211: {1: 0, -1: 0}, -211: {1: 0, -1: 0}},
        "H": {211: {1: 0, -1: 0}, -211: {1: 0, -1: 0}},
        "N": {211: {1: 0, -1: 0}, -211: {1: 0, -1: 0}},
        "C": {211: {1: 0, -1: 0}, -211: {1: 0, -1: 0}},
        "D": {211: {1: 0, -1: 0}, -211: {1: 0, -1: 0}},
    }
    target_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    try:
        while file.nextEvent():
            file.event.read(file.banklist["SIM::Event"])
            file.event.read(file.banklist["REC::Particle"])
            target = int(file.getBytes("SIM::Event", "target")[0])
            target_name = {1: "H", 2: "N", 3: "C", 4: "D"}[target]
            target_counts[target] += 1
            product = int(file.getBytes("SIM::Event", "beam_helicity")[0]) * int(
                file.getBytes("SIM::Event", "target_spin")[0]
            )
            # GiBUU nuclear targets are unpolarized and use target_spin=0;
            # retain them in target_counts but exclude them from polarized
            # asymmetry counters.
            if product not in (-1, 1):
                continue
            for pid in file.getInts("REC::Particle", "pid"):
                if pid in counts["all"]:
                    counts["all"][pid][product] += 1
                    counts[target_name][pid][product] += 1
    finally:
        file.close()

    total = sum(target_counts.values())
    print(f"Target events: H={target_counts[1]}, N={target_counts[2]}, C={target_counts[3]}, D={target_counts[4]}")
    print("target pid    N(product=+)  N(product=-)  raw asymmetry")
    for target_name in ("all", "H", "N", "C", "D"):
        for pid in (211, -211):
            plus = counts[target_name][pid][1]
            minus = counts[target_name][pid][-1]
            raw = (plus - minus) / (plus + minus) if plus + minus else float("nan")
            print(f"{target_name:>6} {pid:>4}   {plus:>12}  {minus:>12}  {raw: .5f}")
    if target_counts[3]:
        print("Carbon is unpolarized, so its raw asymmetry is a background-control check and should be consistent with zero.")
    elif target_counts[4]:
        print("Deuterium is unpolarized in this GiBUU transport sample; use D/N kinematics for the classifier study.")
    else:
        print("The inclusive raw asymmetry is diluted by both P_beam*P_target and the NH3 nitrogen background.")


if __name__ == "__main__":
    main()
