set -ex
apt-get update && apt-get install -y libgl1 libglib2.0-0 curl
cd /workspace/bald-segmentation
pip install -r requirements.txt
until [ -f /workspace/result.tgz ]; do sleep 10; done
tar xzf /workspace/result.tgz
curl -sL -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared
nohup python app.py > /workspace/app.log 2>&1 &
cloudflared tunnel --no-autoupdate --url http://localhost:7860
