#!/usr/bin/env bash
# Runs INSIDE the yutongwu-epio container on one node:  build.sh
# Builds ROCm/mori PR #996 (3a869d5f, the mori of the DO microbenchmarks) in place under src/mori-pr996; every node
# then imports it from NFS (MB_MORI=pr996). UMBP is not needed for EP / IO and stays off; kernels are JIT-compiled on
# first use into MORI_JIT_CACHE_DIR (node-local). One build at a time (lock directory); a tree already built at its
# current commit is only import-checked.
set -uo pipefail
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
MB_MORI=pr996 source "$W/mb/env.sh"
LOG=$W/logs/mori_build_$(hostname -s)_$(date +%m%d%H%M).log
LOCK=$W/src/.mori_build.lock
cd "$MORI_SRC" || exit 1
# the container mounts the real home; keep git's global config out of it
export HOME=/tmp/yt-build-home
mkdir -p "$HOME"
git config --global --add safe.directory "$MORI_SRC" 2>/dev/null || true
while ! mkdir "$LOCK" 2> /dev/null; do
    if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +60 2> /dev/null)" ]; then rmdir "$LOCK" 2> /dev/null; continue; fi
    echo "another build holds $LOCK, waiting"; sleep 20
done
trap 'rmdir "$LOCK" 2> /dev/null' EXIT
check() {
    (cd /tmp && "$PY" -c "
import mori, mori.io
from mori.cco import Communicator
from mori.ops.dispatch_combine_v2 import EpDispatchCombineConfig as C
print('mori', mori.__file__, 'gpu_per_node' in C.__dataclass_fields__)
")
}
head=$(git rev-parse HEAD)
if [ "$(cat "$MORI_SRC/.yt_build_ok" 2> /dev/null)" = "$head" ] && check; then
    echo "BUILD_OK (already built at $head)"; exit 0
fi
echo "=== build $(hostname -s) $(date -u +%FT%TZ) $(git log --oneline -1 | cut -c1-60) with $PY, hipcc $(hipconfig --version 2>/dev/null)"
rm -f "$MORI_SRC/.yt_build_ok"
rm -rf build
for py in "$PY" /usr/bin/python3; do
    if MORI_GPU_ARCHS=gfx950 BUILD_UMBP=OFF BUILD_TESTS=OFF MORI_SKIP_PRECOMPILE=1 "$py" setup.py build_ext > "$LOG" 2>&1; then
        echo "built with $py"; break
    fi
    echo "build with $py failed:"; tail -25 "$LOG"
    [ "$py" = /usr/bin/python3 ] && exit 1
    rm -rf build
done
# build_ext leaves the setuptools extensions (mori.cco.cco needs Cython, from pylib/) under build/lib.*; the EPv2 entry
# imports mori.cco from the source tree
(cd build/lib.*/ && find mori -name '*.cpython-*.so') | while read -r f; do cp build/lib.*/"$f" "python/$f"; done
check || exit 1
strings "$MORI_SRC/python/mori/libmori_ops_v2.so" 2>/dev/null | grep -c MORI_EP_DISABLE_RDMA_ATOMICS
echo "$head" > "$MORI_SRC/.yt_build_ok"
echo "BUILD_OK $(date -u +%FT%TZ)"
