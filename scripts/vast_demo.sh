set -ex
apt-get update && apt-get install -y libgl1 libglib2.0-0 curl
cd /workspace/bald-segmentation
pip install -r requirements.txt
until [ -f /workspace/result.tgz ]; do sleep 10; done
tar xzf /workspace/result.tgz
[ -d examples ] || python -m scripts.fetch_examples
curl -sL -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared
python app.py 2>&1 &
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:7860
