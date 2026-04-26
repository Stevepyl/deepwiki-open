source api/.venv/bin/activate
uv run -m benchmark.run_benchmark \
    --type github \
    --provider openai \
    --model qwen3.6-27b \
    https://github.com/astropy/astropy