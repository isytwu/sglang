import sys, glob, numpy as np
for d in sys.argv[1:]:
    for f in sorted(glob.glob(d + "/io_dev*.npz")):
        z = np.load(f)
        ts, td, nb = z["t_submit"], z["t_done"], z["bytes"]
        dur = td[-1] - ts[0]
        lat = (td - ts) * 1e3
        print(f"{f}: {nb.size} bursts, {nb.sum()/dur/1e9:.1f} GB/s over {dur:.0f} s, burst ms p50 {np.median(lat):.2f} p99 {np.percentile(lat,99):.2f}")
