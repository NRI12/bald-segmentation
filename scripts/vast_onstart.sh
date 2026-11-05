[ -f /workspace/result.tgz ] && exit 0
set -ex
run() { [ -e "$1" ] || { "${@:2}" && touch "$1"; }; }
apt-get update && apt-get install -y libgl1 libglib2.0-0
cd /workspace/bald-segmentation
mkdir -p .done
pip install -r requirements.txt
run data/data.yaml python -m scripts.prepare_data
run data_grade/labels.csv python -m scripts.prepare_grades
run weights/best.pt python -m scripts.train ${TRAIN_ARGS}
run results/metrics_test.json python -m scripts.evaluate
run results/metrics_valid.json python -m scripts.evaluate --split valid
run weights/grader.joblib python -m scripts.fit_grader
run .done/stage1 python -m scripts.train_b --epochs 40 --out weights_b/stage1.pt
run .done/pseudo python -m scripts.pseudo_label
run .done/stage2 python -m scripts.train_b --init weights_b/stage1.pt --pseudo data_grade/pseudo --epochs 20 --lr 3e-5 --out weights_b/stage2.pt
run results/metrics_b_test.json python -m scripts.evaluate_b
run results/metrics_b_valid.json python -m scripts.evaluate_b --split valid
tar czf /workspace/result.tgz weights weights_b results runs/segment/runs/seg/*.png runs/segment/runs/seg/*.csv
echo TRAINING_DONE
