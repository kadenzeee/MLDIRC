#!/usr/bin/env python3

import uproot
import numpy as np
import awkward as ak
import h5py
from tqdm import tqdm
import glob
import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------------------------------------------------------------------------------------------------------------
#
#                                                       FLAGS
#
# --indir       -> Directory of input .root file, generated from .prtdirc macros.
#
# --outdir      -> Directory of output .h5 file. Default = same name as indir, where .root replaced with .h5
#
# --nbins       -> Number of time bins to use. Default = 20
#
# --threads     -> Number of threads to use for shuffling the data
#
# --shufflesize -> Size of blocks to be shuffled. Lower gives more granular shuffling, however increases compute.
#
# ----------------------------------------------------------------------------------------------------------------


def chunk_events(indir, nbins=20, nchan=512, chunk_size=10000):

    with uproot.open(indir) as f:  

        data = f['data']
        theta   = data['PrtEvent/fTof'].array(library='np') * np.pi / 180  # Convert to radians
        phi     = np.zeros_like(theta)
        times       = data['PrtEvent/fHitArray/fHitArray.fLeadTime'].array(library='ak')
        channels    = data['PrtEvent/fHitArray/fHitArray.fChannel'].array(library='ak')
        labels      = data['PrtEvent/fPid'].array(library='ak') 

    assert ak.num(times).tolist() == ak.num(channels).tolist()
    assert len(labels) == len(ak.num(times).tolist())   

    nevents = len(labels)
    print(f"[INFO] Loaded {nevents} events from {indir}") 
    print(os.path.dirname(indir))

    theta = theta + np.random.normal(0, 3E-03, size=len(theta))  # Angle smearing
    phi = phi + np.random.normal(0, 3E-03, size=len(phi))  # Angle smearing 

    angle_features = np.stack([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)], axis=1)    

    nbins = nbins
    time_range = (0, ak.max(times))   # ns

    time_edges =np.linspace(time_range[0], time_range[1], nbins + 1)    

    nchan = nchan
    chan_edges = np.arange(nchan + 1)   

    chunk_size = chunk_size
    n_chunks = int(np.ceil(nevents / chunk_size))   


    chunk_iter = tqdm(range(n_chunks), desc="[CHUNK]", unit="chunk")    

    for chunk_idx in chunk_iter:
        start = chunk_idx * chunk_size
        end = min((chunk_idx + 1) * chunk_size, nevents)

        hists2d = []
        angles = []
        lbls = []   

        for i in range(start, end):
            t = times[i].to_numpy() + np.random.normal(0, 0.2, size=len(times[i]))
            c = channels[i].to_numpy()
            H, _, _ = np.histogram2d(t, c, bins=(time_edges, chan_edges))
            total = H.sum()
            if total > 0:
                H = H / total
            hists2d.append(H.reshape(-1))
            angles.append(angle_features[i])
            lbls.append(labels[i])  

        hists2d = np.stack(hists2d, axis=0)
        angles = np.stack(angles, axis=0)
        lbls = np.array(lbls)   

        

        with h5py.File(f'{os.path.dirname(indir)}/temp_chunk_{chunk_idx}.h5', 'w') as f:
            f.create_dataset('histograms', data=hists2d, compression='gzip')
            f.create_dataset('angles', data=angles, compression='gzip')
            f.create_dataset('labels', data=lbls, compression='gzip')

    chunks_dir = os.path.dirname(indir)

    return chunks_dir



    #def sanity_checks():
#
    #    print("Running sanity checks...")
#
    #    # Use a small random sample
    #    N = 10000
    #    idx = np.random.choice(hists2d.shape[0], N, replace=False)
    #    sample_hist = hists2d[idx]
    #    sample_labels = np.array(labels)[idx]
#
    #    # Check for features that are (almost) identical to the label
    #    for i in range(sample_hist.shape[1]):
    #        corr = np.corrcoef(sample_hist[:, i], sample_labels)[0, 1]
    #        if abs(corr) > 0.99:
    #            print(f"Feature {i} has suspiciously high correlation ({corr:.3f}) with label!")
    #        elif abs(corr) > 0.9:
    #            print(f"Feature {i} has high correlation ({corr:.3f}) with label.")
    #        else:
    #            pass
#
    #    print("Correlation checks done.")    
#
#
    #    # Check if any feature is exactly the label
    #    for i in range(sample_hist.shape[1]):
    #        if np.allclose(sample_hist[:, i], sample_labels):
    #            print(f"Feature {i} is (almost) exactly the label!")
#
    #    print("Exact match checks done.")
#
    #    # Check for duplicate events in the sample
    #    n_unique = len(set(map(tuple, sample_hist)))
    #    if n_unique < N:
    #        print(f"Found {N - n_unique} duplicate events in the sample!")
#
    #    print("Duplicate event checks done.")
#
    #    sample_angle = angle_features[idx]
    #    for i in range(sample_angle.shape[1]):
    #        corr = np.corrcoef(sample_angle[:, i], sample_labels)[0, 1]
    #        if abs(corr) > 0.99:
    #            print(f"Angle feature {i} has suspiciously high correlation ({corr:.3f}) with label!")
#
    #    print("Angle feature correlation checks done.")



def merge_chunks(indir, compression='gzip'):
    files = sorted(glob.glob(os.path.join(indir, 'temp_chunk*.h5')))
    if not files:
        raise FileNotFoundError(f'No chunked files found in {indir}. ')

    outdir = f'{indir}/temp_merged.h5'

    with h5py.File(outdir, 'w', libver='latest') as out:
        out.swmr_mode = True

        total_written = 0

        for i, f in enumerate(files):
            with h5py.File(f, 'r', swmr=True) as src:
                Xh = src['histograms'][:]
                Xa = src['angles'][:]
                Yl = src['labels'][:]

                n_events = len(Yl)
                if n_events == 0:
                    print(f'[WARNING] Skipping {f}: Empty chunks')
                    continue

                if len(Xh) != len(Xa) != len(Yl):
                    print(f'[WARNING] Skipping {f}: Inconsistent data shapes\n-> ({Xh.shape, Xa.shape, Yl.shape})')
                    continue


                if i == 0:                                   
                                
                    out.create_dataset('X_hist',    data=Xh , maxshape=(None,) + Xh.shape[1:], compression=compression, chunks=True)
                    out.create_dataset('X_angles',   data=Xa , maxshape=(None,) + Xa.shape[1:], compression=compression, chunks=True)
                    out.create_dataset('Y_labels',  data=Yl , maxshape=(None,), compression=compression, chunks=True)

                    total_written = n_events
                    print(f'[MERGE] Initialised output file with {n_events} events')

                else:
                  
                    out['X_hist'].resize(total_written + n_events, axis=0)
                    out['X_angles'].resize(total_written + n_events, axis=0)
                    out['Y_labels'].resize(total_written + n_events, axis=0)

                    out['X_hist'][total_written:total_written + n_events] = Xh
                    out['X_angles'][total_written:total_written + n_events] = Xa
                    out['Y_labels'][total_written:total_written + n_events] = Yl

                    total_written += n_events
                    print(f'[MERGE] Appended {n_events} events from {os.path.basename(f)} (Total: {total_written})')

    print(f'[MERGE] Finished merging {len(files)} files')
    print(f'[MERGE] Total events: {total_written}')

    
    
    return outdir, total_written



def shuffle_on_disk(filename, blocksize=1000, n_threads=8, seed=1):
    rng = np.random.default_rng(seed)

    # Read in master file
    with h5py.File(filename, 'r+', libver='latest') as f:
        f.swmr_mode=True
        Xh, Xa, Y = f['X_hist'], f['X_angles'], f['Y_labels']
        n_total = Xh.shape[0]
        n_blocks = n_total // blocksize
        print(f'[INFO] Shuffling {n_total} samples in {n_blocks} blocks')


        # Prepare to shuffle chunks of data
        swaps = rng.integers(0, n_blocks, size=(n_blocks,))

        def swap_blocks(i, j):
            if i == j:
                return
            a = slice(i * blocksize, (i + 1) * blocksize)
            b = slice(j * blocksize, (j + 1) * blocksize)
            tmp_h = Xh[a][:]
            tmp_a = Xa[a][:]
            tmp_y = Y[a][:]
            Xh[a], Xa[a], Y[a] = Xh[b], Xa[b], Y[b]
            Xh[b], Xa[a], Y[b] = tmp_h, tmp_a, tmp_y

        with ThreadPoolExecutor(max_workers=n_threads) as ex:
            futures = [ex.submit(swap_blocks, i, swaps[i]) for i in range(n_blocks)]
            for k, fut in enumerate(as_completed(futures)):
                fut.result()
                if k % 10 == 0:
                    print(f'[SHUFFLE] Progress: {k}/{n_blocks}')
                    f.flush()
    
    print(f'[INFO] On-disk shuffle complete.')



def split_dataset(filename, outdir, train_frac=0.7, val_frac=0.15, compression='gzip'):
    with h5py.File(filename, 'r', swmr=True) as f:
        n_total = f['X_hist'].shape[0]
        n_train = int(train_frac * n_total)
        n_val = int(val_frac * n_total)
        n_test = n_total - n_train - n_val
        print(f'[INFO] Splitting --> train {n_train}, val {n_val}, test {n_test}')

        with h5py.File(outdir, 'w', libver='latest') as fout:
            def copy_slice(name, ds, sl):
                fout.create_dataset(name, data=ds[sl], compression=compression)

            
            copy_slice('Xh_train', f['X_hist'],   slice(0, n_train))
            copy_slice('Xa_train', f['X_angles'], slice(0, n_train))
            copy_slice('Y_train',  f['Y_labels'], slice(0, n_train))

            copy_slice('Xh_val', f['X_hist'],   slice(n_train, n_train+n_val))
            copy_slice('Xa_val', f['X_angles'], slice(n_train, n_train+n_val))
            copy_slice('Y_val',  f['Y_labels'], slice(n_train, n_train+n_val))

            copy_slice('Xh_test', f['X_hist'],   slice(n_train+n_val, n_total))
            copy_slice('Xa_test', f['X_angles'], slice(n_train+n_val, n_total))
            copy_slice('Y_test',  f['Y_labels'], slice(n_train+n_val, n_total))

    print(f'[INFO] Split file written to: {outdir}')


def main():

    parser = argparse.ArgumentParser(description='Merge, shuffle, and split HDF5 chunks with SWMR support.')
    parser.add_argument('--indir', type=str, required=True, help='Input .root file generated from ./prtdirc macros')
    parser.add_argument('--nbins', type=int, default=20, help='Number of time bins for data')
    parser.add_argument('--threads', type=int, default=8, help='Number of threads for shuffling.')
    parser.add_argument('--shufflesize', type=int, default=1000, help='Blocksize for shuffling. Lower integers give more granular shuffling, however may run slower.')
    args = parser.parse_args()
    parser.add_argument('--outdir', type=str, default=os.path.abspath(args.indir).replace('.root', '.h5'), help='Final output file')
    args = parser.parse_args()

    args.indir = os.path.abspath(args.indir)

    chunks_dir = chunk_events(args.indir)
    merged_file, _ = merge_chunks(chunks_dir)
    shuffle_on_disk(merged_file, blocksize=args.shufflesize, n_threads=args.threads)
    split_dataset(merged_file, args.outdir)

    for p in glob.glob(f'{os.path.dirname(args.indir)}/temp*.h5', recursive=True):
        if os.path.isfile(p):
            os.remove(p)

    #if SANITY_CHECKS:
        #sanity_checks()
    
    print('[DONE] All steps complete.')

if __name__ == "__main__":
    main()

