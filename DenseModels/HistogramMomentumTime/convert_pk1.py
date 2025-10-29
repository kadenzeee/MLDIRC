#!/usr/bin/env python3

import os, uproot
import numpy as np
import awkward as ak
import joblib
import time
from tqdm import tqdm
from scipy import sparse
from multiprocessing import Pool

if __name__ == '__main__':

    t1 = time.time()

# ------------------------------------------------
#
#                   PARAMETERS
#
#            To be replaced with flags
#
# ------------------------------------------------

    indir = 'Data/Raw/8K22TO90DEG.root'       # Infile
    nchan = 512                                 # Number of MCP pixels
    aweight = 10                            # Angle weight (this is a multiplier that acts on the raw data)
    asmear = 3E-03                          # Angle smearing in radians
    tsmear = 0.2                            # Time smearing in ns
    tres = 0.1                              # Time bin resolution in ns
    
    trainfrac  = 0.7
    valfrac    = 0.15
    testfrac   = 0.15

# ------------------------------------------------

    with uproot.open(indir) as f:  

        data = f['data']
        theta   = data['PrtEvent/fTof'].array(library='ak') * np.pi / 180  # Convert to radians
        phi     = ak.zeros_like(theta)
        times       = data['PrtEvent/fHitArray/fHitArray.fLeadTime'].array(library='ak')
        channels    = data['PrtEvent/fHitArray/fHitArray.fChannel'].array(library='ak')
        labels      = data['PrtEvent/fPid'].array(library='ak')

        if not times.type.length == channels.type.length == labels.type.length == theta.type.length == phi.type.length:
            raise ValueError('[WARNING] The number of events is not consistent across the dataframe. Regenerate dataset.')

    if trainfrac + valfrac + testfrac != 1.0:
        raise UserWarning('[WARNING] Train-val-test splits do not encompass the full dataset.')

    nevents = labels.type.length

    print(f"[INFO] Loaded {nevents} events from {indir}") 

    theta = theta + np.random.normal(0, asmear, size=len(theta))  # Angle smearing
    phi = phi + np.random.normal(0, asmear, size=len(phi))  # Angle smearing 
    angles = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=1) * aweight

    tmax = np.floor(ak.max(times))
    trange = (0, tmax)            # ns
    nbins = int(tmax / tres)          # time-binning of 100 ps resolution

    print(f"[INFO] Binning into {nbins} time bins with resolution {tres} ns")
    print(f"[INFO] Time smearing at {tsmear} ns, angle smearing at {asmear} rad")

    cedges = np.arange(nchan + 1) # Defining channel bin edges


    # Precompute some histogram binning values
    tmin, tmax = trange[0], trange[-1]
    cmin, cmax = cedges[0], cedges[-1]

    inv_dt = nbins / (tmax - tmin)
    inv_dc = nchan / (cmax - cmin)

    event_iter = tqdm(zip(times, channels), total=len(times), desc="Binning events")        # Creating an iterable for multithreading and tracking progress

    def wrapper(event_iter):
    
        t, c = event_iter

        c = c.to_numpy()
        t = t.to_numpy()

        t += np.random.normal(0, tsmear, size=len(t))

        h = np.zeros((nbins, nchan), dtype=np.float32)

        for i in range(len(t)):
            ix = ((t[i] - tmin) * inv_dt).astype(np.int32)
            iy = ((c[i] - cmin) * inv_dc).astype(np.int32)

            if 0 <= ix < nbins and 0 <= iy < nchan:
                h[ix, iy] += 1

        m = h.max()
        h /= m # Normalize each event  

        return sparse.csr_matrix(h)

    with Pool(os.cpu_count() - 1) as p:
        sparse_events = p.map(wrapper, event_iter)  # Multithread pool. map() preserves the order of events, according to the documentation.

    print(f"[INFO] Stacking events...")

    times = sparse.vstack([sparse.csr_matrix(ev.reshape(1, -1)) for ev in sparse_events]) # times now has shape (nevents, nbins*nchan), and is our main dataset

    print(f"[INFO] Binning done, shuffling and splitting events...cd /")

    shuffle = np.random.permutation(times.shape[0])

    times = times[shuffle]
    angles = angles[shuffle]
    labels = labels[shuffle]

    trainend = int(np.floor(nevents*trainfrac))
    valend = int(trainend + np.floor(nevents*valfrac))

    traintimes  = times[:trainend]
    trainangles = angles[:trainend]
    trainlabels = labels[:trainend]
    valtimes    = times[trainend+1:valend]
    valangles   = angles[trainend+1:valend]
    vallabels   = labels[trainend+1:valend]
    testtimes   = times[valend+1:]
    testangles  = angles[valend+1:]
    testlabels  = labels[valend+1:]

    print(f'[INFO] Saving...')

    outdir = indir.replace('.root', '.pk1')
    joblib.dump({'traintimes': traintimes, 'trainangles':trainangles, 'trainlabels':trainlabels, 
                 'valtimes':valtimes, 'valangles':valangles, 'vallabels':vallabels, 
                 'testtimes':testtimes, 'testangles':testangles, 'testlabels':testlabels}, outdir)

    t2 = time.time()

    print(f'[INFO] Done in {t2-t1} seconds')
