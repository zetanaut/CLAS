#!/usr/bin/env python3
"""Generate a small, deterministic simulated CLAS12 HIPO file."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import hipopy


def generate(path: Path, events: int = 100, seed: int = 12345) -> None:
    """Write ``events`` simulated events to *path*."""
    if events < 1:
        raise ValueError("events must be positive")

    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    output = hipopy.create(str(path))

    # The schema is stored inside the HIPO file, so the file is self-describing.
    output.newTree(
        "REC::Event",
        {"event": "I", "npart": "S", "weight": "F"},
        group=300,
        item=1,
    )
    output.newTree(
        "REC::Particle",
        {"pid": "I", "px": "F", "py": "F", "pz": "F", "charge": "B"},
        group=300,
        item=2,
    )
    output.open()

    try:
        for event_number in range(events):
            n_particles = int(rng.integers(1, 7))
            pid = rng.choice(np.array([11, 211, -211, 2212]), size=n_particles)
            momentum = rng.normal(0.0, 1.0, size=(n_particles, 3)).astype(np.float32)
            momentum[:, 2] = np.abs(momentum[:, 2]) + 0.5
            charge = np.where(pid == 11, -1, np.where(pid == -211, -1, 1))

            # hipopy expects arrays with shape (columns, rows).
            output.writeBank(
                "REC::Event",
                ["event", "npart", "weight"],
                [
                    np.array([event_number]),
                    np.array([n_particles]),
                    np.array([rng.uniform(0.8, 1.2)], dtype=np.float32),
                ],
                dtypes="ISF",
            )
            output.writeBank(
                "REC::Particle",
                ["pid", "px", "py", "pz", "charge"],
                [pid, momentum[:, 0], momentum[:, 1], momentum[:, 2], charge],
                dtypes="IFFFB",
            )
            output.addEvent()
            output.event.reset()
    finally:
        output.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("simulated.hipo"))
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()
    generate(args.output, args.events, args.seed)
    print(f"Wrote {args.events} events to {args.output}")


if __name__ == "__main__":
    main()
