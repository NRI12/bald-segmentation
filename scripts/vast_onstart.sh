[ -f /workspace/result.tgz ] && exit 0
set -ex
apt-get update && apt-get install -y libgl1 libglib2.0-0
cd /workspace/bald-segmentation
pip install -r requirements.txt
python -m scripts.prepare_data
python -m scripts.prepare_grades
python -m scripts.train ${TRAIN_ARGS}
python -m scripts.evaluate
python -m scripts.evaluate --split valid
python -m scripts.fit_grader
python -m scripts.train_b --epochs 40 --out weights_b/stage1.pt
python -m scripts.pseudo_label
python -m scripts.train_b --init weights_b/stage1.pt --pseudo data_grade/pseudo --epochs 20 --lr 3e-5 --out weights_b/stage2.pt
python -m scripts.evaluate_b
python -m scripts.evaluate_b --split valid
tar czf /workspace/result.tgz weights weights_b results runs/segment/runs/seg/*.png runs/segment/runs/seg/*.csv
echo TRAINING_DONE
