set -ex
apt-get update && apt-get install -y libgl1 libglib2.0-0
cd /workspace/bald-segmentation
pip install -r requirements.txt
python -m scripts.prepare_data
python -m scripts.train ${TRAIN_ARGS}
python -m scripts.evaluate
python -m scripts.evaluate --split valid
tar czf /workspace/result.tgz weights results runs/segment/runs/seg/*.png runs/segment/runs/seg/*.csv
echo TRAINING_DONE
