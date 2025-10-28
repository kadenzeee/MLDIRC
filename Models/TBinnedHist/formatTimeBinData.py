import uproot
import numpy as np
import awkward as ak
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt

SHOW_PROGRESS = True

f = uproot.open('ANGULAR_DATASETS/70DEG.root')

data = f['data']

times       = data['PrtEvent/fHitArray/fHitArray.fLeadTime'].array(library='ak')
channels    = data['PrtEvent/fHitArray/fHitArray.fChannel'].array(library='ak')
labels      = data['PrtEvent/fPid'].array(library='ak')

assert ak.num(times).tolist() == ak.num(channels).tolist()
assert len(labels) == len(ak.num(times).tolist())

nbins = 20
time_range = (0, 50)   # ns
time_edges =np.linspace(time_range[0], time_range[1], nbins + 1)

nchan = 512
chan_edges = np.arange(nchan + 1)

hists2d = []

event_iter = zip(times, channels)
if SHOW_PROGRESS:
    event_iter = tqdm(event_iter, total=len(times), desc="Binning events")

for t, c in event_iter:
    H, _, _ = np.histogram2d(t.to_numpy(), c.to_numpy(),
                             bins = (time_edges, chan_edges))
    total = H.sum()
    if total > 0:
        H = H / total # Normalize each event

    hists2d.append(H)

hists2d = np.stack(hists2d, axis=0)

X = hists2d.reshape(hists2d.shape[0], -1)
Y = labels

with h5py.File('TBinnedHist/70DEG.h5', 'w') as f:
    f.create_dataset(
        'X', data=X,
        compression='gzip',
        chunks=(1, X.shape[1])
    )
    f.create_dataset('Y', data=Y)

H = X[-1].reshape(nbins, nchan)
for i in range(63):
    H += X[i].reshape(nbins, nchan)
    
plt.figure(figsize=(10, 6))
plt.imshow(H.T, aspect='auto', origin='lower',
           extent=(0, nbins, 0, nchan))
plt.colorbar(label='Counts')
plt.xlabel('Time bin')
plt.ylabel('Channel')
plt.title(f'Event {i}, Label: {Y[i]}')
plt.show()
