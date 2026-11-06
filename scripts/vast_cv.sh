set -ex
apt-get update && apt-get install -y libgl1 libglib2.0-0
cd /workspace/bald-segmentation
pip install --retries 10 --timeout 60 -r requirements.txt
python -m scripts.prepare_data
python -m scripts.prepare_grades
python -m scripts.cv_grades ${CV_ARGS}
echo CV_DONE
