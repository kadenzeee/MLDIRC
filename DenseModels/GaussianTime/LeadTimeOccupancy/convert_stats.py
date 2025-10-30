import sys
import ROOT # type: ignore ::: need to be in environment with ROOT installed and sourced
import numpy as np
import time

program_start = time.time()

ROOT.gInterpreter.ProcessLine('#include "/u/markhoff/Documents/Simulations/prttools/PrtTools.h"')
ROOT.gSystem.Load('/u/markhoff/Documents/Simulations/prtdirc/build/libPrt.so')

infile = "GaussianTime/BinaryOccupancy/80DEG.root"
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

TIMES  = np.zeros((entries, nchan+4))
LABELS = np.zeros(entries)

while t.next() and t.i() < entries:

    if not bool(t.event().getHits()): # Skips empty events
        continue

    i = t.i()

    times = [photon.getLeadTime() + ROOT.gRandom.Gaus(0, 0.2) for photon in t.event().getHits()]
    chs   = [int(photon.getChannel()) for photon in t.event().getHits()]

    mu    = np.mean(times)
    std   = np.std(times)
    t0    = np.min(times)
    t1    = np.max(times)

    chind = np.zeros(nchan)
    chind[chs] += times[chs]

    TIMES[i]  = np.concatenate(([mu, std, t0, t1], chind))
    LABELS[i] = t.pid()/2 - 1


shuffle = np.random.permutation(entries)

TIMES  = TIMES[shuffle]
LABELS = LABELS[shuffle]

outfile = infile.replace(".root", "")
np.savez(outfile, TIMES=TIMES, LABELS=LABELS)

program_end = time.time()
print(f"Done in {program_end-program_start} seconds")
