#!/usr/bin/env bash
# Runs INSIDE the yutongwu-epio container:  [MB_MORI=pr996|image] [IOTC=..] py.sh <script> <args...>  (log to stdout)
W=/apps/yutongwu/store-vllm/k31-vllm-ep-io-smci355-20261119
source "$W/mb/env.sh" || exit 2
cd /tmp
exec "$PY" "$@"
