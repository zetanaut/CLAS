# Python HIPO sandbox

This workspace uses [`hipopy`](https://pypi.org/project/hipopy/) to create and
read CLAS12 HIPO files.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The environment contains `hipopy`, NumPy/Matplotlib, scikit-learn, and
PyTorch. On this machine the working virtual environment is `.venv`; use its
Python explicitly in commands below.

## What has been built in this directory

This is a staged Run Group C-inspired SIDIS study, not yet a full detector
simulation. The current levels are:

1. A small Python HIPO generator for learning the file format.
2. A smeared SIDIS toy with reconstructed electrons and charged pions.
3. clasDIS-based proton/neutron LUND input bridged into HIPO.
4. GiBUU D, N, and C transport samples bridged into HIPO.
5. Normalized distribution plots, cross-section-weighted yield plots, and
   logistic-regression/PyTorch target classifiers.

The full `GiBUU → GEMC → COATJAVA → reconstructed HIPO` chain is not installed
yet. GEMC/Geant4 and COATJAVA are the next detector-level upgrade.

## Generate and inspect a file

```bash
./.venv/bin/python generate_hipo.py simulated.hipo --events 1000 --seed 12345
./.venv/bin/python read_hipo.py simulated.hipo
```

The generator writes self-describing `REC::Event` and `REC::Particle` banks.
The particle multiplicity, PID, momentum, and charge are simulated with a
reproducible NumPy random generator.

## Run Group C-inspired SIDIS toy

```bash
./.venv/bin/python generate_rgc_sidis.py rgc_sidis.hipo --events 10000 --seed 12345
./.venv/bin/python analyze_rgc_sidis.py rgc_sidis.hipo
./.venv/bin/python make_rgc_plots.py rgc_sidis.hipo
```

Add `--target carbon` to generate a pure-carbon control sample, or use
`--target both` to create both `rgc_sidis_nh3.hipo` and
`rgc_sidis_carbon.hipo`. The event bank tags free hydrogen as `target=1`,
nitrogen as `target=2`, and carbon as `target=3`. The plotter produces
normalized H/N/C/inclusive distributions for `x`, `Q2`, `W`, `y`, `z`, `pT`,
`phi_h`, pion momentum, and pion angle.

For an educational H-versus-N baseline classifier, use:

```bash
./.venv/bin/python train_target_classifier.py rgc_classifier.hipo
```

The classifier excludes the truth target label and uses only reconstructed
kinematics, missing mass, and vertex observables. Its score measures separation
in this toy model; it should not be interpreted as a demonstrated event-by-event
hydrogen/nitrogen tagger for real data.

The reconstructed banks now include a momentum-dependent forward-track smear
(about 1% at 5 GeV), approximately 1 mrad polar-angle and 4 mrad azimuthal
smearing, vertex smearing, six-sector coil gaps, and simple electron/pion
tracking and PID efficiencies. Reconstructed `x`, `Q2`, `W`, `y`, `z`, `pT`,
and `phi_h` are recalculated from the smeared tracks and stored in `REC::Event`;
the generator-level values remain in `SIM::Event`. `REC::Event` also contains
the reconstructed `vz` and the missing-mass observables `mm2_e_pi` and
`mm_e_pi` for the detected `e'pi` system. For the inclusive/SIDIS
configuration, the target vertex is sampled from a 5 cm longitudinal cell and
a 1.9 cm diameter transverse beam raster. The smaller 1.4 cm raster applies to
the exclusive Forward-Tagger-in configuration, not this SIDIS toy.

Geometry audit: published CLAS12 specifications support the 5--35 degree
forward charged-track range, approximately 1% momentum resolution at 5 GeV,
and milliradian-scale angular resolution. The six-sector structure is also
documented. The exact sector dead-zone width and run-dependent target-coordinate
offset were not available as universal public numbers, so the current 3-degree
sector-gap cut and zero target-coordinate offset remain explicit toy assumptions,
not surveyed detector geometry.

## Higher-fidelity physics input with clasDIS

For the optional higher-fidelity workflow, clone Jefferson Lab's
[`clasDIS`](https://github.com/JeffersonLab/clasdis) into `external/clasdis`.
It is a PEPSI-LUND generator with PDF-based DIS cross sections, polarized-proton
input, fragmentation, and a CLAS12 acceptance option. On this macOS machine it can be built with the
isolated compiler environment created below:

```bash
/Users/dustin/miniforge3/bin/conda create -y -n clasdis-build -c conda-forge gfortran
cd external/clasdis
/Users/dustin/miniforge3/bin/conda run -n clasdis-build make FOR=-lgfortran
mkdir -p eventfiles
CLASDIS_PDF=$PWD/pdf ./clasdis --trig 10000 --nmax 10000 \
  --beam 10.6 --q 1 10 --w 4 50 --z 0.3 --zpos 0 --zwidth 5 \
  --raster 1.9 --targ proton --pol 80 --accclas12 --pid 211
```

The generated LUND files can be bridged into HIPO with detector smearing and
the RGC vertex model. Supplying proton and neutron files makes the nitrogen
and carbon samples use separate elementary inputs rather than relabeling every
event as a proton:

```bash
cd ../..
.venv/bin/python convert_clasdis_lund.py \
  external/clasdis/eventfiles/*.0000.dat rgc_clasdis_nuclear_nh3.hipo \
  --material nh3 --limit 5000
```

This is a meaningful upgrade for the proton component: the `x`, `Q2`, `W`,
`z`, transverse momentum, azimuthal structure, and pion multiplicity come from
the generator rather than independent Python draws. Nitrogen and carbon remain
effective nuclear mixtures of proton/neutron clasDIS events with Fermi-like
missing-mass broadening. A genuine nuclear calculation would require a nuclear
event generator/transport model such as GiBUU and a full GEMC/coatjava detector
chain. The resulting H-versus-N classifier is intentionally harder and should
be treated as a study benchmark, not a measured separation performance.

The official CLAS12 software workflow is `clasDIS → GEMC → evio2hipo →
coatjava`. GEMC and the CLAS12 software image are container-oriented, while
coatjava requires Java; this macOS workspace currently lacks both Docker and a
Java runtime, so that detector-level step is staged rather than silently
approximated.

## GiBUU nuclear transport upgrade

The optional GiBUU 2023 source tree is expected at `external/gibuu` and should
be obtained under its distribution terms; it is not included in this demo
repository. It was built locally with the same isolated GNU Fortran toolchain
used for clasDIS. The source includes a
CLAS/JLab high-energy lepton mode explicitly optimized for 10.6 GeV and target
cards for C, N, and D. The helper script creates transport cards with a
configurable ensemble count and final perturbative-particle output. A small
test run is:

```bash
.venv/bin/python make_gibuu_rgc_jobcards.py --ensembles 200
cd external/gibuu/testRun
PATH=/Users/dustin/miniforge3/envs/gibuu-build/bin:$PATH \
  ./GiBUU.x < jobCards/rgc/rgc_C.job > /tmp/gibuu_C.log 2>&1
PATH=/Users/dustin/miniforge3/envs/gibuu-build/bin:$PATH \
  ./GiBUU.x < jobCards/rgc/rgc_N.job > /tmp/gibuu_N.log 2>&1
PATH=/Users/dustin/miniforge3/envs/gibuu-build/bin:$PATH \
  ./GiBUU.x < jobCards/rgc/rgc_D.job > /tmp/gibuu_D.log 2>&1
cd ../../..
```

The output bridge is:

```bash
.venv/bin/python convert_gibuu_lhe.py \
  external/gibuu/testRun/EventOutput.Pert.00000001.lhe \
  rgc_gibuu_nitrogen.hipo --target nitrogen --limit 500
```

GiBUU’s high-energy perturbative file stores transported hadrons and event
metadata (`nu`, `Q2`, azimuth, and event type), but not the scattered electron
four-vector. The bridge reconstructs the electron from those exact lepton
kinematics before applying the detector-resolution model. This is preferable to
inventing an independent electron distribution, but it should be validated
against a future detector-level chain.

The generated [rgc_gibuu_carbon.hipo](rgc_gibuu_carbon.hipo),
[rgc_gibuu_nitrogen.hipo](rgc_gibuu_nitrogen.hipo), and
[rgc_gibuu_deuterium.hipo](rgc_gibuu_deuterium.hipo) contain GiBUU-transported
charged pions. For an ND3-like study, merge three deuterium blocks with one
nitrogen block and classify target code 4 (D) against target code 2 (N):

```bash
.venv/bin/python train_target_classifier.py rgc_gibuu_nd3.hipo \
  --positive-target 4 --negative-target 2
```

The current [rgc_gibuu_nd3.hipo](rgc_gibuu_nd3.hipo) is a controlled mock
mixture with the same GiBUU generator and detector bridge for D and N, so its
classifier is not dominated by a generator-domain mismatch. GiBUU's shadowing
routine cannot be used for A≤2, so the deuterium card explicitly disables that
optional correction. The transport sample is also unpolarized; it is suitable
for the D/N kinematic classifier, but not for a final spin-asymmetry study.

For a higher-statistics comparison, the 2,000-ensemble D and N runs are
available as [rgc_gibuu_deuterium_full.hipo](rgc_gibuu_deuterium_full.hipo)
and [rgc_gibuu_nitrogen_2000.hipo](rgc_gibuu_nitrogen_2000.hipo). The
cross-section-weighted, unnormalized comparison is produced with:

```bash
.venv/bin/python make_rgc_yield_plots.py \
  rgc_gibuu_deuterium_full.hipo rgc_gibuu_nitrogen_2000.hipo
```

This uses the GiBUU event weights and multiplies the deuterium contribution by
three for the three deuterons in ND3. The resulting yields are in
generator-weight units rather than detector-luminosity-normalized counts.

The PyTorch classifier example is:

```bash
.venv/bin/python train_target_classifier_torch.py \
  rgc_gibuu_dn_independent.hipo --positive-target 4 --negative-target 2
```

Use the independent one-copy D/N sample for evaluation; the composition-sized
ND3 mixture repeats blocks and can otherwise leak duplicate events between
training and test splits.

## Current artifacts and quick checks

Useful generated files include:

- [rgc_gibuu_nd3_highstat.hipo](rgc_gibuu_nd3_highstat.hipo): composition-sized
  ND3-like mock sample.
- [rgc_gibuu_nd3_highstat_distributions.png](rgc_gibuu_nd3_highstat_distributions.png):
  normalized D/N distributions.
- [rgc_gibuu_nd3_yields.png](rgc_gibuu_nd3_yields.png): cross-section-weighted,
  unnormalized yields.
- [rgc_gibuu_dn_independent.hipo](rgc_gibuu_dn_independent.hipo): one-copy
  independent D/N sample for classifier evaluation.

Basic checks:

```bash
.venv/bin/python read_hipo.py rgc_gibuu_nd3_highstat.hipo
.venv/bin/python analyze_rgc_sidis.py rgc_gibuu_nd3_highstat.hipo
MPLCONFIGDIR=.matplotlib MPLBACKEND=Agg .venv/bin/python make_rgc_plots.py \
  rgc_gibuu_nd3_highstat.hipo --output rgc_gibuu_nd3_highstat_distributions.png
MPLCONFIGDIR=.matplotlib MPLBACKEND=Agg .venv/bin/python make_rgc_yield_plots.py \
  rgc_gibuu_deuterium_full.hipo rgc_gibuu_nitrogen_2000.hipo
```

The current PyTorch example reports about AUC 0.55 on the independent D/N
sample. The higher score obtained from the composition-sized sample should not
be used as a performance claim because repeated blocks can leak duplicate
events across a random train/test split.
