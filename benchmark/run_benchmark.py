"""Benchmark runner for DeepWiki agent.

Runs questions from benchmark/ground_truth/{repo_name}.jsonl through the
DeepWiki agent and writes results to:
  benchmark/output/raw_results/{repo_name}.jsonl  -- full event traces
  benchmark/output/eval/{repo_name}.jsonl          -- question+answer pairs for judge

Usage (from project root, with api venv activated):
    source api/.venv/bin/activate
    python -m benchmark.run_benchmark <repo_url> [options]

Options:
    --type      github | gitlab | bitbucket | local  (default: github)
    --token     access token for private repos
    --provider  LLM provider name (default: from api/config/generator.json)
    --model     model name (default: provider default)
    --language  output language code, e.g. en / zh  (default: en)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ── path bootstrap ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

# Suppress INFO logs from api.* modules — setup_logging() attaches a console
# handler to the root logger at import time; silence it for CLI use.
import logging  # noqa: E402
import warnings  # noqa: E402
import io  # noqa: E402
logging.disable(logging.INFO)
warnings.filterwarnings("ignore")

# ── api imports ───────────────────────────────────────────────────────────────
# Suppress noisy print()/FutureWarning output from third-party libs at import time.
_real_stderr, sys.stderr = sys.stderr, io.StringIO()
from api.agent import (  # noqa: E402
    AgentMessage,
    ErrorEvent,
    FinishEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    UnifiedProvider,
    get_agent_config,
    get_tools_for_agent,
    run_agent_loop,
)
from api.agent.wiki_generator import _ADALFLOW_ROOT, _LANGUAGE_NAMES  # noqa: E402
from api.config import configs  # noqa: E402
from api.data_pipeline import DatabaseManager  # noqa: E402
sys.stderr = _real_stderr

# ── paths ─────────────────────────────────────────────────────────────────────
_BENCHMARK_DIR = Path(__file__).resolve().parent
_GROUND_TRUTH_DIR = _BENCHMARK_DIR / "ground_truth"
_RAW_RESULTS_DIR = _BENCHMARK_DIR / "output" / "raw_results"
_EVAL_DIR = _BENCHMARK_DIR / "candidate"

_SLUG_RE = re.compile(r"[^a-z0-9]+")


# =============================================================================
# Section 1: CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="benchmark.run_benchmark",
        description="Run DeepWiki agent against benchmark ground truth questions.",
    )
    ap.add_argument("repo_url", help="GitHub / GitLab / Bitbucket URL or local path")
    ap.add_argument(
        "--type",
        dest="repo_type",
        default="github",
        choices=["github", "gitlab", "bitbucket", "local"],
    )
    ap.add_argument("--token", default=None, metavar="PAT",
                    help="Access token for private repos")
    ap.add_argument("--provider", default=None,
                    help="LLM provider (overrides generator.json default)")
    ap.add_argument("--model", default=None,
                    help="Model name (overrides provider default)")
    ap.add_argument(
        "--language",
        default="en",
        choices=sorted(_LANGUAGE_NAMES.keys()),
        metavar=f"LANG ({'/'.join(sorted(_LANGUAGE_NAMES.keys()))})",
        help="Output language code (default: en)",
    )
    return ap.parse_args()


def resolve_provider_model(args: argparse.Namespace) -> tuple[str, str]:
    provider = args.provider or configs.get("default_provider", "google")
    providers_cfg: dict = configs.get("providers", {})
    if provider not in providers_cfg:
        sys.exit(
            f"Unknown provider '{provider}'. "
            f"Available: {sorted(providers_cfg.keys())}"
        )
    model = args.model or providers_cfg[provider].get("default_model", "")
    if not model:
        sys.exit(
            f"No model specified and provider '{provider}' has no default_model "
            f"in api/config/generator.json"
        )
    return provider, model


# =============================================================================
# Section 2: Repo helpers
# =============================================================================


def extract_ground_truth_name(repo_url: str) -> str:
    """Extract repo name for ground truth file matching (last URL segment)."""
    return repo_url.rstrip("/").split("/")[-1].replace(".git", "")


def compute_repo_meta(repo_url: str, repo_type: str) -> tuple[str, str]:
    """Return (repo_name, repo_path) for agent use.

    repo_name uses owner_repo format (e.g. django_django).
    repo_path is the local filesystem path.
    """
    if repo_type == "local":
        repo_path = os.path.abspath(os.path.expanduser(repo_url))
        repo_name = os.path.basename(repo_path.rstrip(os.sep)) or "local-repo"
        return repo_name, repo_path

    parts = repo_url.rstrip("/").split("/")
    if repo_type in ("github", "gitlab", "bitbucket") and len(parts) >= 5:
        owner = parts[-2]
        repo = parts[-1].replace(".git", "")
        repo_name = f"{owner}_{repo}"
    else:
        repo_name = parts[-1].replace(".git", "")
    repo_path = os.path.join(_ADALFLOW_ROOT, "repos", repo_name)
    return repo_name, repo_path


# =============================================================================
# Section 3: Ground truth I/O
# =============================================================================


def load_ground_truth(gt_name: str) -> list[dict[str, str]]:
    """Load questions from benchmark/ground_truth/{gt_name}.jsonl."""
    gt_path = _GROUND_TRUTH_DIR / f"{gt_name}.jsonl"
    if not gt_path.exists():
        available = sorted(p.stem for p in _GROUND_TRUTH_DIR.glob("*.jsonl"))
        sys.exit(
            f"Ground truth file not found: {gt_path}\n"
            f"Available repos: {available}"
        )
    records = []
    with open(gt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def count_existing_results(output_path: Path) -> int:
    """Count valid JSON lines in output file for resume support."""
    if not output_path.exists():
        return 0
    count = 0
    with open(output_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError:
                pass
    return count


def _truncate_jsonl(path: Path, keep_lines: int) -> None:
    """Rewrite path keeping only the first keep_lines valid JSON lines."""
    if not path.exists():
        return
    kept: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    json.loads(line)
                    kept.append(line if line.endswith("\n") else line + "\n")
                except json.JSONDecodeError:
                    pass
            if len(kept) >= keep_lines:
                break
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(kept)


def reset_repo_state(repo_path: str) -> None:
    """Restore repo working tree to HEAD state after agent may have mutated it.

    Uses checkout + clean rather than reset --hard to avoid touching HEAD/index.
    Silently ignores errors (e.g. non-git local repos).
    """
    subprocess.run(["git", "checkout", "--", "."], cwd=repo_path, capture_output=True)
    subprocess.run(["git", "clean", "-fd"], cwd=repo_path, capture_output=True)


# =============================================================================
# Section 4: Agent runner
# =============================================================================


async def process_question(
    question: str,
    agent_config: Any,
    provider: UnifiedProvider,
    tools: dict,
    repo_path: str,
    prompt_vars: dict[str, str],
) -> dict:
    """Run a single question through the agent loop and collect the full trace.

    Returns a dict with keys: question, steps, answer, error, metadata.
    Each step = one ReAct iteration: {text, tool_calls}.
    A new step begins when TextDelta arrives after a ToolCallEnd.
    """
    messages = [AgentMessage.user(question)]

    steps: list[dict] = []
    current_text: list[str] = []
    current_tool_calls: list[dict] = []
    pending_args: dict[str, dict] = {}  # tool_call_id -> tool_args
    after_tool_end = False
    finish_reason = "unknown"
    error: str | None = None
    start_ms = time.monotonic()

    def _flush_step() -> None:
        text = "".join(current_text)
        if text or current_tool_calls:
            steps.append({"text": text, "tool_calls": list(current_tool_calls)})
        current_text.clear()
        current_tool_calls.clear()

    try:
        async for evt in run_agent_loop(
            agent_config, messages, provider, tools, repo_path, prompt_vars
        ):
            if isinstance(evt, TextDelta):
                if after_tool_end:
                    _flush_step()
                    after_tool_end = False
                current_text.append(evt.content)

            elif isinstance(evt, ToolCallStart):
                pending_args[evt.tool_call_id] = evt.tool_args

            elif isinstance(evt, ToolCallEnd):
                current_tool_calls.append({
                    "name": evt.tool_name,
                    "args": pending_args.pop(evt.tool_call_id, {}),
                    "result": evt.result_summary,
                    "is_error": evt.is_error,
                    "duration_ms": evt.duration_ms,
                })
                after_tool_end = True

            elif isinstance(evt, FinishEvent):
                finish_reason = evt.finish_reason

            elif isinstance(evt, ErrorEvent):
                error = evt.error

    except Exception as exc:
        error = str(exc)

    _flush_step()

    # Final answer = text from the last step that has text (the concluding response)
    answer = ""
    for step in reversed(steps):
        if step["text"]:
            answer = step["text"]
            break

    total_ms = int((time.monotonic() - start_ms) * 1000)

    return {
        "question": question,
        "steps": steps,
        "answer": answer,
        "error": error,
        "metadata": {
            "finish_reason": finish_reason,
            "total_duration_ms": total_ms,
            "total_steps": len(steps),
        },
    }


# =============================================================================
# Section 5: Orchestrator
# =============================================================================


async def run_benchmark(args: argparse.Namespace) -> None:
    provider_name, model = resolve_provider_model(args)
    gt_name = extract_ground_truth_name(args.repo_url)
    agent_repo_name, repo_path = compute_repo_meta(args.repo_url, args.repo_type)
    language_name = _LANGUAGE_NAMES.get(args.language, args.language)

    print(f"Provider  : {provider_name}/{model}")
    print(f"Repo      : {args.repo_url}")
    print(f"GT file   : {gt_name}.jsonl")
    print(f"Language  : {language_name}")

    print("\nCloning / verifying repo...")
    DatabaseManager()._create_repo(
        repo_url_or_path=args.repo_url,
        repo_type=args.repo_type,
        access_token=args.token,
    )

    ground_truth = load_ground_truth(gt_name)
    total = len(ground_truth)

    _RAW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = _RAW_RESULTS_DIR / f"{gt_name}.jsonl"
    eval_path = _EVAL_DIR / f"{gt_name}.jsonl"

    skip = count_existing_results(raw_path)
    if skip:
        eval_count = count_existing_results(eval_path)
        skip = min(skip, eval_count)
        _truncate_jsonl(raw_path, skip)
        _truncate_jsonl(eval_path, skip)
        print(f"\nResuming: {skip}/{total} already done, skipping.")

    agent_config = get_agent_config("wiki")
    tools = get_tools_for_agent(agent_config, repo_path)
    provider = UnifiedProvider(provider_name, model)
    prompt_vars = {
        "repo_type": args.repo_type,
        "repo_url": args.repo_url,
        "repo_name": agent_repo_name,
        "language_name": language_name,
    }

    with open(raw_path, "a", encoding="utf-8") as raw_f, \
         open(eval_path, "a", encoding="utf-8") as eval_f:

        for i, record in enumerate(ground_truth):
            if i < skip:
                continue

            question = record["question"]
            print(f"\n[{i + 1}/{total}] {question[:100]}{'...' if len(question) > 100 else ''}")

            try:
                result = await process_question(
                    question, agent_config, provider, tools, repo_path, prompt_vars
                )
            except Exception as exc:
                result = {
                    "question": question,
                    "steps": [],
                    "answer": "",
                    "error": str(exc),
                    "metadata": {"finish_reason": "error", "total_duration_ms": 0, "total_steps": 0},
                }
            finally:
                reset_repo_state(repo_path)

            raw_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            raw_f.flush()

            if result["error"] is None:
                eval_record = {"question": question, "answer": result["answer"]}
                eval_f.write(json.dumps(eval_record, ensure_ascii=False) + "\n")
                eval_f.flush()

            status = f"error: {result['error']}" if result["error"] else f"{result['metadata']['total_duration_ms']}ms"
            print(f"  -> {status}")

    print(f"\nDone: {total} questions")
    print(f"  raw  -> {raw_path}")
    print(f"  eval -> {eval_path}")


def main() -> None:
    args = parse_args()
    asyncio.run(run_benchmark(args))


if __name__ == "__main__":
    main()
