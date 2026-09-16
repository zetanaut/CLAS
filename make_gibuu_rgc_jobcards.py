#!/usr/bin/env python3
"""Create small GiBUU 2023 Run-Group-C-like C and N job cards."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def make_card(template: Path, output: Path, target_z: int, target_a: int, input_path: Path, ensembles: int, shadow: bool = True) -> None:
    text = template.read_text()
    text = re.sub(r"(?m)^      numEnsembles\s*=.*$", f"      numEnsembles    =         {ensembles}", text)
    text = re.sub(r"(?m)^      path_To_Input\s*=.*$", f"      path_To_Input   = '{input_path.resolve()}'", text)
    text = re.sub(r"(?m)^      iExperiment=13 ! Hermes, 27GeV, arXiv:0704.3712 \(pT-broadening\)$",
                  "      iExperiment=19 ! CLAS/JLAB, 12GeV Run Group A optimized 10.6 GeV", text)
    text = text.replace("&HiLeptonNucleus\n", f"&HiLeptonNucleus\n      shadow = {'T' if shadow else 'F'}\n", 1)
    text = re.sub(r"(?m)^!      DoLeptonKinematics = T$", "      DoLeptonKinematics = T", text)
    text = re.sub(r"(?m)^!      DoHadronKinematics = T$", "      DoHadronKinematics = T", text)
    text = re.sub(r"(?m)^!      DoOutChannels = T$", "      DoOutChannels = T", text)
    text = re.sub(r"(?m)^!\s+WritePerturbativeParticles = T.*$", "      WritePerturbativeParticles = T", text)
    text = re.sub(r"(?m)^!\s+EventFormat=1.*$", "      EventFormat=1", text)
    text = re.sub(r"(?m)^      Z=  6, A= 12 ! C$", f"      Z= {target_z}, A= {target_a}", text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gibuu-root", type=Path, default=Path("external/gibuu"))
    parser.add_argument("--ensembles", type=int, default=200)
    args = parser.parse_args()
    template = args.gibuu_root / "testRun/jobCards/014_HiLepton_A.job"
    output_dir = args.gibuu_root / "testRun/jobCards/rgc"
    make_card(template, output_dir / "rgc_C.job", 6, 12, args.gibuu_root / "buuinput", args.ensembles)
    make_card(template, output_dir / "rgc_N.job", 7, 14, args.gibuu_root / "buuinput", args.ensembles)
    make_card(template, output_dir / "rgc_H.job", 1, 1, args.gibuu_root / "buuinput", args.ensembles)
    make_card(template, output_dir / "rgc_D.job", 1, 2, args.gibuu_root / "buuinput", args.ensembles, shadow=False)
    print(f"Wrote {output_dir / 'rgc_C.job'}, {output_dir / 'rgc_N.job'}, {output_dir / 'rgc_H.job'}, and {output_dir / 'rgc_D.job'}")


if __name__ == "__main__":
    main()
