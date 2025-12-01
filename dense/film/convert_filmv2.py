import sys
import os
import ROOT  # type: ignore ::: need to be in environment with ROOT installed and sourced
import numpy as np
import time
import matplotlib.pyplot as plt 
from scipy import sparse

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

binslength = 0.1 # ns
eventlength = 30 # ns

# -------------------------------------


trainend = int(np.floor(entries*trainfrac))
valend   = int(trainend + np.floor(entries*valfrac))

TIMES  = np.zeros((entries, nchan, 2))
ANGLES = np.zeros((entries, 7))
LABELS = np.zeros(entries)

nbins = eventlength // binslength 

print(f"Binning into {nbins} bins")

while t.next() and t.i() < entries:

    if not bool(t.event().getHits()): # Skips empty events
        continue

    i = t.i()

    # gathering data
    times = [photon.getLeadTime() + ROOT.gRandom.Gaus(0, 0.2) for photon in t.event().getHits()]
    chs   = [int(photon.getChannel()) for photon in t.event().getHits()]
    theta = t.event().getTof() + ROOT.gRandom.Gaus(0, 3E-03)
    phi   = t.event().getTofP() + ROOT.gRandom.Gaus(0, 3E-03)
    
    # statistics for angular PID
    mu    = np.mean(times)
    std   = np.std(times)
    t0    = np.min(times)
    t1    = np.max(times)

    # ignore hits after {eventlength} ns
    times = [t for t in times if (t < eventlength or t > 0)]

    tbins = [int(10*t) for t in times]
    tbins = [tbin % nbins for tbin in tbins]
    
    chind = np.zeros(nchan)
    chind[chs] += chs
    
    tind  = np.zeros(nchan)
    tind[chs]  += tbins 
    
    event = np.vstack((chind, tind))
    event = np.reshape()

    TIMES[i]  = event
    ANGLES[i] = [mu, std, t0, t1, np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)]
    LABELS[i] = t.pid()/2 - 1


shuffle = np.random.permutation(entries)

TIMES  = TIMES[shuffle]
ANGLES = ANGLES[shuffle]
LABELS = LABELS[shuffle]

print('[INFO] Shuffle ok')

outfile = infile.replace(".root", "")
outfile = os.path.basename(f'{outfile}v2')
np.savez_compressed(outfile, TIMES=TIMES, ANGLES=ANGLES, LABELS=LABELS)

print('[INFO] Save ok')

program_end = time.time()
print(f"Done in {program_end-program_start} seconds")
