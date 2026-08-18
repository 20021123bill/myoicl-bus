#!/usr/bin/env bash
set -uo pipefail
cd /data2/chenyuxiang/code/myoicl
export PATH=/data2/chenyuxiang/conda_envs/qwerty/bin:$PATH
mkdir -p /tmp/v41 bus/jobs
tar xzf tools/myoicl_v41_drop.tar.gz -C /tmp/v41
python /tmp/v41/patch_v41.py .
rc=$?
if [ "$rc" -ne 0 ]; then
  echo "PATCH FAILED rc=$rc -- refusing to queue D1/D2"
  exit "$rc"
fi
cp /tmp/v41/bus/jobs/010_archive.sh      bus/jobs/
cp /tmp/v41/bus/jobs/020_d1_gatefix.sh   bus/jobs/
cp /tmp/v41/bus/jobs/030_d2_forcectx.sh  bus/jobs/
echo "v4.1 applied; 010/020/030 queued for the next runner cycle"
