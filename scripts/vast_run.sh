set -ex
apt-get update && apt-get install -y libgl1 libglib2.0-0 curl
cd /workspace/bald-segmentation
pip install --retries 10 --timeout 60 -r requirements.txt
if [ ! -f /workspace/bald-segmentation-topdown.tgz ]; then
  python -m scripts.prepare_data
  python -m scripts.prepare_topdown
  python -m scripts.train
  python -m scripts.evaluate
  python -m scripts.fetch_examples
  rm -rf raw_examples
  tar czf /workspace/bald-segmentation-topdown.tgz -C /workspace bald-segmentation
fi
echo RUN_DONE
curl -sL -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared
python -u app.py --share 2>&1 &
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:7860
