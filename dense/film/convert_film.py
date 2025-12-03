#!/usr/bin/env python3

import sys
import os
import ROOT  # type: ignore ::: need to be in environment with ROOT installed and sourced
import numpy as np
import time

program_start = time.time()

ROOT.gInterpreter.ProcessLine('#include "../../../prttools/PrtTools.h"')
ROOT.gSystem.Load("../../../prtdirc/build/libPrt.dylib")

infile = "../../../Data/Raw/8K22TO90DEG.root"
if(len(sys.argv) > 1):
    infile = sys.argv[1] 

t = ROOT.PrtTools(infile)
entries = t.entries()
nchan = t.npix() * t.npmt()

# -------------------------------------
#
#               SPLITS
#
# -------------------------------------

trainfrac  = 0.7
valfrac    = 0.15
testfrac   = 0.15

# -------------------------------------


trainend = int(np.floor(entries*trainfrac))
valend   = int(trainend + np.floor(entries*valfrac))

TIMES  = np.zeros((entries, nchan))
ANGLES = np.zeros((entries, 7))
LABELS = np.zeros(entries)

while t.next() and t.i() < entries:

    if not bool(t.event().getHits()): # Skips empty events
        continue

    i = t.i()

    times = [photon.getLeadTime() + ROOT.gRandom.Gaus(0, 0.2) for photon in t.event().getHits()]
    chs   = [int(photon.getChannel()) for photon in t.event().getHits()]
    theta = t.event().getTof()
    phi   = t.event().getTofP()

    mu    = np.mean(times)
    std   = np.std(times)
    t0    = np.min(times)
    t1    = np.max(times)

    chind = np.zeros(nchan)
    chind[chs] += 1

    TIMES[i]  = chind
    ANGLES[i] = [mu, std, t0, t1, np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    LABELS[i] = t.pid()/2 - 1


shuffle = np.random.permutation(entries)

TIMES  = TIMES[shuffle]
ANGLES = ANGLES[shuffle]
LABELS = LABELS[shuffle]

outfile = infile.replace(".root", "")
outfile = os.path.basename(outfile)
np.savez_compressed(outfile, TIMES=TIMES, ANGLES=ANGLES, LABELS=LABELS)

program_end = time.time()
print(f"Done in {program_end-program_start} seconds")
