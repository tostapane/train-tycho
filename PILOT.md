# Brev pilot for Tycho’s 22 tools

- **Size and budget.** Start with 300 recipe attempts and up to 150 of the kept conversations per teacher. **50,000 passing conversations is the later production target.** In Brev, pick one stoppable RTX PRO 6000 with 96 GB VRAM and at least 200 GB disk. Record the hourly price and start time. Suggested pilot cap: $5 of the $60. Set a timer for slightly less than `60 × 5 / hourly_price` minutes to leave time for backup. Setup and downloads use billed time too.

- **Laptop, from this repo.** Set `GPU` to the working SSH alias from Brev. Copy the current files, including uncommitted fixes, to a fresh pilot directory. Keep this terminal open for backup.

  ```bash
  GPU='your-brev-ssh-alias'
  ssh "$GPU" 'mkdir -p ~/workspace/tycho-pilot'
  rsync -a --exclude=__pycache__/ app gen cases *.py pyproject.toml uv.lock .python-version "$GPU:workspace/tycho-pilot/"
  ssh -t "$GPU" 'tmux new-session -A -s tycho'
  ```

- **GPU, inside tmux.** Run each command in Bash and check it succeeded before entering the next. Paste function definitions as a unit. Use the same shell throughout. The Brev image needs Docker with NVIDIA GPU access, tmux, rsync and curl.

  ```bash
  cd ~/workspace/tycho-pilot
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  uv run --locked python -m unittest
  nvidia-smi
  docker pull vllm/vllm-openai:cu130-nightly
  IMAGE=$(docker image inspect -f '{{.Id}}' vllm/vllm-openai:cu130-nightly)
  mkdir -p ~/workspace/hf-cache out
  printf '%s\n' "$IMAGE" > out/serving-image.txt
  ```

- **Start the 27B teacher.** Serving automatically downloads the official FP8 weights into the shared cache. The two models are **Qwen/Qwen3.5-27B-FP8** and **Qwen/Qwen3.5-35B-A3B-FP8**. Load one at a time. The helper waits for the API and prints the served model ID. If startup fails, it prints the container log. [27B checkpoint](https://huggingface.co/Qwen/Qwen3.5-27B-FP8) · [35B checkpoint](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8) · [vLLM Blackwell setup](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)

  ```bash
  serve() {
    docker run -d --name "$1" --gpus all --ipc=host -p 127.0.0.1:8080:8000 \
      -v "$HOME/workspace/hf-cache:/root/.cache/huggingface" "$IMAGE" "$2" \
      --language-model-only --enable-prefix-caching --max-model-len 8192 \
      --max-num-seqs 64 --gpu-memory-utilization 0.92 || return 1
    until curl -fs http://127.0.0.1:8080/v1/models; do
      if [[ $(docker inspect -f '{{.State.Running}}' "$1") != true ]]; then
        docker logs --tail 40 "$1"; return 1
      fi
      sleep 5
    done
  }
  serve tycho-pilot-27b Qwen/Qwen3.5-27B-FP8
  uv run --locked smoke.py
  ```

  In smoke output, look for the correct model ID, a successful completion, thinking that stopped, and a pass-2 run. A host-side warning about vLLM or cached weights is expected with Docker. If smoke reports a failure or cannot reach the teacher, resolve it before generating.

- **27B writes, then plays.** Each generation command **stops by itself**. After writing succeeds, copy the folder **once, before either teacher plays**, so both receive identical messages. The helper counts already saved runs and also works after an interruption.

  ```bash
  play150() {
    local saved=0
    if [[ -f "$1/runs.jsonl" ]]; then saved=$(wc -l < "$1/runs.jsonl"); fi
    if (( saved < 150 )); then
      uv run --locked generate.py runs --count "$((150 - saved))" --workers 64 --out "$1"
    fi
  }
  uv run --locked generate.py messages --count 300 --workers 64 --out out/pilot-27b
  test ! -e out/pilot-35b && cp -a out/pilot-27b out/pilot-35b
  play150 out/pilot-27b
  ```

- **Switch teachers manually.** Once the 27B batch finishes, stop its server, start the 35B, and repeat the smoke check. Play the copied cases first. The final command separately compares the 35B’s writing on the same 300 recipe seeds. Stop the server when finished.

  ```bash
  docker stop tycho-pilot-27b
  serve tycho-pilot-35b Qwen/Qwen3.5-35B-A3B-FP8
  uv run --locked smoke.py
  play150 out/pilot-35b
  uv run --locked generate.py messages --count 300 --workers 64 --out out/pilot-35b-write
  docker stop tycho-pilot-35b
  ```

- **If interrupted or near the cap.** Press **Ctrl-C once** in the generator and wait for the shell prompt. In-flight requests can take time to finish. Completed rows are flushed to disk. A printed case-level `FAIL` is a scored model result. For a retryable server error, restore the same teacher and rerun the interrupted `messages` command or `play150` call in the same folder. Skip the folder-copy step on resume. After an SSH disconnect, reconnect with `ssh -t "$GPU" 'tmux attach -t tycho'`. To restart an already-created container, use `docker start tycho-pilot-27b` or `docker start tycho-pilot-35b` and wait for `/v1/models`.

- **Laptop again. Back up, then stop the Brev instance.** Use a second local terminal with the same `GPU` value, or detach tmux with **Ctrl-B, then D**. Copy and verify even a partial pilot. In the Brev console click **Stop** and confirm **Stopped**. Closing SSH or stopping Docker leaves the GPU billed. Storage charges can continue while stopped. [Brev billing](https://docs.nvidia.com/brev/concepts/gpu-instances)

  ```bash
  mkdir -p out/brev-pilot
  rsync -av "$GPU:workspace/tycho-pilot/out/" out/brev-pilot/
  rsync -acn --itemize-changes "$GPU:workspace/tycho-pilot/out/" out/brev-pilot/
  ```

  The verification command should print no changed files. After stopping the instance, review locally:

  ```bash
  for d in out/brev-pilot/pilot-*; do uv run stats.py "$d"; done
  less out/brev-pilot/pilot-35b/cases.csv
  ```

  Inspect matching seeds in both teachers' `runs.jsonl` files for the messages, tool arguments and answers. Read accepted and rejected examples, especially lookups, dates and follow-ups. Compare correct conversations per billed dollar and coverage by kind. Use the measured yield to choose the production count and teachers before starting another paid run.
