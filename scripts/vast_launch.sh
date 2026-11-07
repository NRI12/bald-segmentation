#!/bin/bash
set -e
onstart=$(mktemp)
bundle=$(git archive HEAD ${BUNDLE:-app.py baldseg scripts weights/norwood_thresholds.json requirements.txt} | gzip -9 | base64 -w0)
echo "mkdir -p /workspace/bald-segmentation && echo $bundle | base64 -d | tar xz -C /workspace/bald-segmentation" > "$onstart"
cat "${ONSTART:-scripts/vast_run.sh}" >> "$onstart"
vastai create instance "$1" --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime --disk 30 --ssh --label bald-seg \
  --env "-e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY -e TRAIN_ARGS='${TRAIN_ARGS:-}'" --onstart "$onstart"
