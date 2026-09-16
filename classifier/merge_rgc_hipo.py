#!/usr/bin/env python3
"""Merge HIPO files produced by the RGC converters into one sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import hipopy
import numpy as np


SIM_FLOATS = ("x", "q2", "w", "y", "z", "phi_h", "p_t", "a_ll", "weight", "vx", "vy", "vz")
REC_EVENT_FLOATS = ("x", "q2", "w", "y", "z", "phi_h", "p_t", "mm2_e_pi", "mm_e_pi", "vz")


def merge(inputs: list[Path], output_path: Path) -> int:
    as_array = lambda values: np.asarray(values)
    out = hipopy.create(str(output_path))
    out.newTree("SIM::Event", {"event": "I", "target": "B", "beam_helicity": "B", "target_spin": "B", "npart": "S", **{name: "F" for name in SIM_FLOATS}}, group=1000, item=1)
    out.newTree("MC::Particle", {"pid": "I", "px": "F", "py": "F", "pz": "F", "charge": "B"}, group=1000, item=2)
    out.newTree("REC::Particle", {"pid": "I", "px": "F", "py": "F", "pz": "F", "charge": "B", "status": "S", "vx": "F", "vy": "F", "vz": "F"}, group=300, item=1)
    out.newTree("REC::Event", {"event": "I", **{name: "F" for name in REC_EVENT_FLOATS}}, group=300, item=2)
    out.open()
    event_number = 0
    try:
        for path in inputs:
            src = hipopy.open(str(path))
            for bank in ("SIM::Event", "MC::Particle", "REC::Particle", "REC::Event"):
                src.readBank(bank)
            try:
                while src.nextEvent():
                    for bank in ("SIM::Event", "MC::Particle", "REC::Particle", "REC::Event"):
                        src.event.read(src.banklist[bank])
                    sim_names = ["event", "target", "beam_helicity", "target_spin", "npart"] + list(SIM_FLOATS)
                    sim_values = [np.array([event_number]), *[as_array(src.getBytes("SIM::Event", name)) for name in ("target", "beam_helicity", "target_spin")], as_array(src.getShorts("SIM::Event", "npart")), *[as_array(src.getFloats("SIM::Event", name)) for name in SIM_FLOATS]]
                    out.writeBank("SIM::Event", sim_names, sim_values, dtypes="IBBBS" + "F" * len(SIM_FLOATS))
                    pids = src.getInts("MC::Particle", "pid")
                    out.writeBank("MC::Particle", ["pid", "px", "py", "pz", "charge"], [as_array(pids), as_array(src.getFloats("MC::Particle", "px")), as_array(src.getFloats("MC::Particle", "py")), as_array(src.getFloats("MC::Particle", "pz")), as_array(src.getBytes("MC::Particle", "charge"))], dtypes="IFFFB")
                    out.writeBank("REC::Particle", ["pid", "px", "py", "pz", "charge", "status", "vx", "vy", "vz"], [as_array(src.getInts("REC::Particle", "pid")), as_array(src.getFloats("REC::Particle", "px")), as_array(src.getFloats("REC::Particle", "py")), as_array(src.getFloats("REC::Particle", "pz")), as_array(src.getBytes("REC::Particle", "charge")), as_array(src.getShorts("REC::Particle", "status")), as_array(src.getFloats("REC::Particle", "vx")), as_array(src.getFloats("REC::Particle", "vy")), as_array(src.getFloats("REC::Particle", "vz"))], dtypes="IFFFBSFFF")
                    rec_names = ["event"] + list(REC_EVENT_FLOATS)
                    rec_values = [np.array([event_number]), *[as_array(src.getFloats("REC::Event", name)) for name in REC_EVENT_FLOATS]]
                    out.writeBank("REC::Event", rec_names, rec_values, dtypes="I" + "F" * len(REC_EVENT_FLOATS))
                    out.addEvent()
                    out.event.reset()
                    event_number += 1
            finally:
                src.close()
    finally:
        out.close()
    return event_number


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(f"Merged {merge(args.inputs, args.output)} events into {args.output}")


if __name__ == "__main__":
    main()
