#!/usr/bin/env bash
# Runs ON a smci355 host:  rate.sh [secs]  ->  per-device RDMA tx / rx GB/s of ionic_0..7 over the window. Read-only.
s=${1:-2}
rd() { for i in 0 1 2 3 4 5 6 7; do cat /sys/class/infiniband/ionic_$i/ports/1/hw_counters/{tx,rx}_rdma_ucast_bytes; done | tr '\n' ' '; }
a=$(rd); sleep "$s"; b=$(rd)
python3 -c "
import sys
a = list(map(int, sys.argv[1].split())); b = list(map(int, sys.argv[2].split())); s = float(sys.argv[3])
tx = [(b[2 * i] - a[2 * i]) / s / 1e9 for i in range(8)]; rx = [(b[2 * i + 1] - a[2 * i + 1]) / s / 1e9 for i in range(8)]
print('tx', ' '.join(f'{v:5.1f}' for v in tx), '| rx', ' '.join(f'{v:5.1f}' for v in rx), '| sum tx %.1f rx %.1f GB/s' % (sum(tx), sum(rx)))
" "$a" "$b" "$s"
