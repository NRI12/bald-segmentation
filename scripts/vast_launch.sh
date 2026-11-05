#!/bin/bash
set -e
onstart=$(mktemp)
bundle=$(git archive HEAD baldseg scripts requirements.txt | gzip -9 | base64 -w0)
echo "mkdir -p /workspace/bald-segmentation && echo $bundle | base64 -d | tar xz -C /workspace/bald-segmentation" > "$onstart"
cat scripts/vast_onstart.sh >> "$onstart"
if [ -n "$UPDATE" ]; then
  vastai update instance "$UPDATE" --onstart "$(cat "$onstart")"
  exit
fi
vastai create instance "$1" --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --disk 30 --ssh --label bald-seg \
  --env "-e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY -e TRAIN_ARGS='${TRAIN_ARGS:-}'" --onstart "$onstart"
