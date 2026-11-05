set -ex
apt-get update && apt-get install -y libgl1 libglib2.0-0 curl
cd /workspace/bald-segmentation
pip install --retries 10 --timeout 60 -r requirements.txt
until [ -f /workspace/copy.done ]; do sleep 10; done
cp -rn /workspace/src/bald-segmentation/. /workspace/bald-segmentation/
if [ ! -f results/norwood/metrics_b_valid.json ]; then
  python -m scripts.fit_grader
  python -m scripts.train_b --init weights_b/stage2.pt --pseudo data_grade/pseudo --epochs 15 --lr 3e-5 --batch 4 --out weights_b/norwood.pt
  python -m scripts.evaluate_b --weights weights_b/norwood.pt --out results/norwood
  python -m scripts.evaluate_b --weights weights_b/norwood.pt --split valid --out results/norwood
  tar czf /workspace/result_norwood.tgz weights/grader.joblib weights_b/norwood.pt results
fi
echo NORWOOD_DONE
[ -d examples ] || python -m scripts.fetch_examples
curl -sL -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared
python app.py 2>&1 &
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:7860
