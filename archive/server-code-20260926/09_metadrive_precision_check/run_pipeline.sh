#!/usr/bin/env bash
set -Eeuo pipefail
BASE=/root/autodl-tmp/metadrive_precision_20260923
cd "$BASE"
# A single invocation only; timeout/error is never automatically resumed.
exec 9>checks/pipeline.lock
flock -n 9 || { echo 'PIPELINE_STOPPED another precision pipeline holds the lock'; exit 70; }
if [[ -e checks/LAUNCH_ONCE || -e calibration || -e confirmation ]]; then
 echo 'PIPELINE_STOPPED existing launch/data; manual inspection required, automatic resume forbidden'; exit 71
fi
sha256sum -c SOURCE_PROTOCOL_SHA256SUMS.txt > checks/prelaunch_sha256.log 2>&1
mkdir checks/LAUNCH_ONCE
date -u +%FT%TZ > checks/LAUNCH_ONCE/started_utc.txt
set +e
timeout --signal=TERM --kill-after=30s 9h bash run_pipeline_inner.sh
rc=$?
set -e
printf '%s\n' "$rc" > checks/LAUNCH_ONCE/exit_code.txt
date -u +%FT%TZ > checks/LAUNCH_ONCE/ended_utc.txt
if [[ "$rc" == 124 || "$rc" == 137 ]]; then
 echo "PIPELINE_STOPPED wallclock9h_limit rc=$rc; partial files retained; no automatic resume"
elif [[ "$rc" != 0 ]]; then
 echo "PIPELINE_STOPPED nonzero_exit rc=$rc; partial files retained; no automatic resume"
fi
exit "$rc"
