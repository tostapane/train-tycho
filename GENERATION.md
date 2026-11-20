# Generating the training data on Brev

The 27B does both passes: it writes the user messages, then plays them on the fake phone (80-87% pass in the pilot).
One command runs both, in the background; every case is saved as soon as it's done, and both passes resume where
they stopped.

**Sizing.** `N` recipes give about `0.73 × N` passing cases (14% of messages dropped, ~15% of runs fail).
Pilot speed on one RTX PRO 6000: each 10,000 recipes take about 15 min of messages and 1.4 h of runs.

| N | time | passing cases |
|---|---|---|
| 30,000 | ~5 h | ~22k |
| 40,000 | ~7 h | ~29k |
| 50,000 | ~8.5 h | ~37k |

Pick `N` so the run ends around when you wake up: the instance bills while idle too.

## 1. Instance and code

RTX PRO 6000 (96 GB), 200 GB disk. In the web terminal (or JupyterLab → Terminal):

```bash
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -fsSL https://astral.sh/uv/install.sh | sh
cd "$HOME/workspace/tycho-pilot-web" &&   # the pilot's checkout, cloned with the deploy key
  git -c core.sshCommand="ssh -i $HOME/.ssh/tycho_brev -o IdentitiesOnly=yes" pull && git log -1 --oneline
uv run --locked python -m unittest
```

New instance: clone with a read-only deploy key (`ssh-keygen -t ed25519`, then GitHub → Settings → Deploy keys),
`git clone git@github.com:tostapane/train-tycho.git`, and `docker pull vllm/vllm-openai:cu130-nightly`.

## 2. Start the 27B

`--restart unless-stopped` brings vLLM back by itself if it crashes in the night.

```bash
docker rm -f tycho-web-27b teacher 2>/dev/null
docker run -d --name teacher --restart unless-stopped --gpus all --ipc=host -p 127.0.0.1:8080:8000 \
  -v "$HOME/workspace/hf-cache:/root/.cache/huggingface" vllm/vllm-openai:cu130-nightly Qwen/Qwen3.5-27B-FP8 \
  --language-model-only --enable-prefix-caching --max-model-len 8192 \
  --max-num-seqs 128 --gpu-memory-utilization 0.92
until curl -fs http://127.0.0.1:8080/v1/models >/dev/null; do sleep 10; done; echo ready
```

Loading takes a few minutes (`docker logs -f teacher` shows it; the pilot already downloaded the weights).

## 3. Smoke, then launch and sleep

```bash
uv run --locked smoke.py
```

Part 4 plays 8 cases through the real generator and reads the saved files back: no FAIL lines, then launch.

```bash
N=50000
nohup bash -c "
for try in 1 2 3 4 5 6; do uv run --locked generate.py messages --count $N --workers 128 --out out/data-brev && break; sleep 120; done
for try in 1 2 3 4 5 6; do uv run --locked generate.py runs --count 1000000 --workers 128 --out out/data-brev && break; sleep 120; done
" > out/generation.log 2>&1 &
tail -f out/generation.log    # Ctrl+C stops watching; the run goes on, even if the browser closes
```

Watch a few minutes: lines of messages, then (after the messages pass) lines of runs with PASS/FAIL. Each loop
retries if a batch stops on errors (e.g. vLLM restarting): failed cases aren't saved, so they run again.
Progress: `wc -l out/data-brev/*.jsonl`, or `uv run --locked stats.py out/data-brev` for pass rates.

## 4. Morning: save and stop

```bash
docker logs teacher > out/data-brev/vllm.log 2>&1
tar -czf ../tycho-data.tar.gz -C out data-brev && ls -lh ../tycho-data.tar.gz
```

Download it with JupyterLab (`jupyter lab --no-browser --ip=0.0.0.0 --port=8888 --ServerApp.root_dir="$HOME/workspace"`,
then Brev → Access → Using Tunnels → port 8888; right-click the file → Download). On the laptop:
`gzip -t ~/Downloads/tycho-data.tar.gz`, then **Stop** the instance in Brev: it bills until stopped.
