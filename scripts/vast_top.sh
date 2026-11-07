set -ex
apt-get update && apt-get install -y libgl1 libglib2.0-0 curl
cd /workspace/bald-segmentation
pip install --retries 10 --timeout 60 -r requirements.txt
until [ -f /workspace/src/bald-segmentation/weights_b/norwood.pt ]; do sleep 10; done
while a=$(du -s /workspace/src); sleep 60; [ "$a" != "$(du -s /workspace/src)" ]; do :; done
cp -rn /workspace/src/bald-segmentation/. .
sed -i "s#^path: .*#path: $PWD/data#" data/data.yaml
[ -f results/top/metrics_top.json ] || python -m scripts.evaluate_top
echo TOP_DONE
curl -sL -o /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x /usr/local/bin/cloudflared
python -u app.py --share 2>&1 &
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:7860
