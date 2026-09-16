#!/usr/bin/env python3
"""Read back a generated HIPO file and print a short summary."""

import argparse

import hipopy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    args = parser.parse_args()

    file = hipopy.open(args.input)
    try:
        print("Banks:", ", ".join(file.getBanks()))
        file.readBank("REC::Event")
        file.readBank("REC::Particle")
        count = 0
        while file.nextEvent():
            # hipopybind first loads the raw event, then each bank is decoded
            # from that event into its reusable Bank object.
            file.event.read(file.banklist["REC::Event"])
            file.event.read(file.banklist["REC::Particle"])
            if count == 0:
                print("event:", file.getInts("REC::Event", "event"))
                print("particle pid:", file.getInts("REC::Particle", "pid"))
                print("particle px:", file.getFloats("REC::Particle", "px"))
            count += 1
        print("Events:", count)
    finally:
        file.close()


if __name__ == "__main__":
    main()
