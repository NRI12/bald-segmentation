#!/bin/bash
set -ex
apt-get update && apt-get install -y git libgl1 libglib2.0-0
cd /workspace
git clone -b "${BRANCH:-main}" https://github.com/NRI12/bald-segmentation.git
cd bald-segmentation
pip install -r requirements.txt
python -m scripts.prepare_data
python -m scripts.train ${TRAIN_ARGS}
python -m scripts.evaluate
python -m scripts.evaluate --split valid
tar czf /workspace/result.tgz weights results runs/segment/runs/seg/*.png runs/segment/runs/seg/*.csv
echo TRAINING_DONE
