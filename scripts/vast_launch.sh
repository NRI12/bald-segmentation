#!/bin/bash
set -e
onstart=$(mktemp)
bundle=$(git archive HEAD ${BUNDLE:-app.py baldseg scripts requirements.txt} | xz -9e | base64 -w0)
echo "mkdir -p /workspace/bald-segmentation && echo $bundle | base64 -d | python -c 'import io,sys,tarfile; tarfile.open(fileobj=io.BytesIO(sys.stdin.buffer.read()),mode=\"r:xz\").extractall(\"/workspace/bald-segmentation\")'" > "$onstart"
cat "${ONSTART:-scripts/vast_onstart.sh}" >> "$onstart"
vastai create instance "$1" --image pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime --disk 30 --ssh --label bald-seg \
  --env "-e ROBOFLOW_API_KEY=$ROBOFLOW_API_KEY -e TRAIN_ARGS='${TRAIN_ARGS:-}'" --onstart "$onstart"
