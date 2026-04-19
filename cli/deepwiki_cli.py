"""DeepWiki Agent CLI demo — single-file entry point.

Usage (from project root, with api venv activated):
    source api/.venv/bin/activate
    python -m cli.deepwiki_cli <repo_url> [options]

Options:
    --type          github | gitlab | bitbucket | local  (default: github)
    --token         access token for private repos
    --provider      LLM provider name (default: from api/config/generator.json)
    --model         model name (default: provider default)
    --language      output language code, e.g. en / zh / ja  (default: en)
    --comprehensive generate 15-20 page wiki with sections tree
                    (default: concise 4-6 page flat list)

REPL commands:
    /wiki           run two-phase wiki generation; pages saved to
                    ~/.adalflow/wiki-output/<repo_name>/
    /clear          reset Q&A conversation history
    /exit  Ctrl+C   quit
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any

# ─── path bootstrap ───────────────────────────────────────────────────────────
# Must happen before api imports so `import api.*` resolves from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Load .env before api imports — some provider clients read env vars at import time.
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

# ─── api imports ──────────────────────────────────────────────────────────────
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
from api.agent.wiki_generator import (  # noqa: E402
    _ADALFLOW_ROOT,
    _COMPREHENSIVE_INSTRUCTION,
    _CONCISE_INSTRUCTION,
    _LANGUAGE_NAMES,
    _flatten_pages_in_section_order,
    _format_planner_user_prompt,
    _format_writer_user_prompt,
    parse_wiki_structure,
)
from api.config import configs  # noqa: E402
from api.data_pipeline import DatabaseManager  # noqa: E402
from api.utils.repo_tree import build_file_tree, read_repo_readme  # noqa: E402

# ─── constants ────────────────────────────────────────────────────────────────
WIKI_OUTPUT_ROOT = Path(_ADALFLOW_ROOT) / "wiki-output"
_TOOL_ARG_PREVIEW = 80


# =============================================================================
# Section 1: Bootstrap
# =============================================================================


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="deepwiki-cli",
        description="DeepWiki agent CLI: multi-turn Q&A and two-phase wiki generation.",
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
        help="Wiki output language code (default: en)",
    )
    ap.add_argument(
        "--comprehensive",
        action="store_true",
        help="Generate 8-12 page wiki with sections tree (default: 4-6 page concise)",
    )
    return ap.parse_args()


def resolve_provider_model(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve provider and model from CLI args + generator.json config."""
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
# Section 2: Repo metadata helpers
# =============================================================================

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def compute_repo_meta(repo_url: str, repo_type: str) -> tuple[str, str]:
    """Return (repo_name, repo_path).

    For local repos, repo_path is the user-supplied absolute path — DatabaseManager
    keeps local repos at the original filesystem location (no copy to ~/.adalflow/repos/).
    For remote repos, mirrors wiki_generator._compute_repo_path logic.
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


def slugify(title: str, max_len: int = 60) -> str:
    s = _SLUG_RE.sub("-", title.lower()).strip("-")
    return (s or "page")[:max_len]


# =============================================================================
# Section 3: Event printing
# =============================================================================


def _format_tool_args(args_dict: dict[str, Any]) -> str:
    if not args_dict:
        return ""
    pairs: list[str] = []
    for k, v in args_dict.items():
        s = v if isinstance(v, str) else repr(v)
        if len(s) > _TOOL_ARG_PREVIEW:
            s = s[:_TOOL_ARG_PREVIEW] + "..."
        pairs.append(f"{k}={s}")
    return ", ".join(pairs)


def print_text_delta(content: str) -> None:
    print(content, end="", flush=True)


def print_tool_call(name: str, args_dict: dict[str, Any], indent: str = "") -> None:
    print(f"\n{indent}[{name}({_format_tool_args(args_dict)})]", flush=True)


def print_tool_end(end: ToolCallEnd, indent: str = "") -> None:
    status = "ERR" if end.is_error else "OK"
    print(f"{indent}  └─ {status} ({end.duration_ms}ms)", flush=True)


# =============================================================================
# Section 4: Prompt vars builders
# =============================================================================


def qa_prompt_vars(
    repo_url: str, repo_type: str, repo_name: str, language: str
) -> dict[str, str]:
    return {
        "repo_type": repo_type,
        "repo_url": repo_url,
        "repo_name": repo_name,
        "language_name": _LANGUAGE_NAMES.get(language, language),
    }


def planner_prompt_vars(
    repo_url: str,
    repo_type: str,
    repo_name: str,
    language: str,
    comprehensive: bool,
) -> dict[str, str]:
    return {
        **qa_prompt_vars(repo_url, repo_type, repo_name, language),
        "comprehensive_instruction": (
            _COMPREHENSIVE_INSTRUCTION if comprehensive else _CONCISE_INSTRUCTION
        ),
    }


# =============================================================================
# Section 5: Async handlers
# =============================================================================


async def run_qa_turn(
    query: str,
    history: list[AgentMessage],
    provider: UnifiedProvider,
    agent_config: Any,
    tools: dict,
    repo_path: str,
    prompt_vars: dict[str, str],
) -> None:
    """Stream one Q&A turn to stdout; append both sides to history in-place.

    run_agent_loop does not mutate the caller's messages list — the CLI is
    responsible for appending user and assistant turns for multi-turn continuity.
    """
    history.append(AgentMessage.user(query))
    chunks: list[str] = []
    async for evt in run_agent_loop(
        agent_config, history, provider, tools, repo_path, prompt_vars
    ):
        if isinstance(evt, TextDelta):
            chunks.append(evt.content)
            print_text_delta(evt.content)
        elif isinstance(evt, ToolCallStart):
            print_tool_call(evt.tool_name, evt.tool_args)
        elif isinstance(evt, ToolCallEnd):
            print_tool_end(evt)
        elif isinstance(evt, ErrorEvent):
            print(f"\n[error] {evt.error} (code={evt.code})", file=sys.stderr)
        elif isinstance(evt, FinishEvent):
            print()
    history.append(AgentMessage.assistant_text("".join(chunks)))


async def run_planner(
    provider: UnifiedProvider,
    repo_path: str,
    repo_url: str,
    repo_type: str,
    repo_name: str,
    language: str,
    comprehensive: bool,
) -> dict | None:
    """Phase 1: run wiki-planner; return parsed wiki structure dict or None."""
    cfg = get_agent_config("wiki-planner")
    tools = get_tools_for_agent(cfg, repo_path)
    file_tree = build_file_tree(repo_path)
    readme = read_repo_readme(repo_path)
    user_prompt = _format_planner_user_prompt(file_tree, readme, comprehensive, language)
    messages = [AgentMessage.user(user_prompt)]
    raw_chunks: list[str] = []

    print("\n=== Phase 1: planning ===", flush=True)
    async for evt in run_agent_loop(
        cfg,
        messages,
        provider,
        tools,
        repo_path,
        planner_prompt_vars(repo_url, repo_type, repo_name, language, comprehensive),
    ):
        if isinstance(evt, TextDelta):
            raw_chunks.append(evt.content)  # JSON — collect silently, don't stream
        elif isinstance(evt, ToolCallStart):
            print_tool_call(evt.tool_name, evt.tool_args)
        elif isinstance(evt, ToolCallEnd):
            print_tool_end(evt)
        elif isinstance(evt, ErrorEvent):
            print(f"\n[planner error] {evt.error}", file=sys.stderr)

    structure = parse_wiki_structure("".join(raw_chunks))
    if structure is None:
        print(
            "\n[error] planner output could not be parsed as a wiki structure JSON",
            file=sys.stderr,
        )
        return None

    pages = _flatten_pages_in_section_order(structure)
    print(f"\nWiki structure: {len(pages)} pages")
    for i, p in enumerate(pages, 1):
        print(f"  {i:>2}. {p['title']}")
    return structure


async def run_writer_for_page(
    provider: UnifiedProvider,
    repo_path: str,
    repo_url: str,
    repo_type: str,
    repo_name: str,
    language: str,
    page: dict,
    page_index: int,
    total_pages: int,
    output_dir: Path,
) -> None:
    """Phase 2 (one page): stream markdown to stdout and save to .md file."""
    cfg = get_agent_config("wiki-writer")
    tools = get_tools_for_agent(cfg, repo_path)
    user_prompt = _format_writer_user_prompt(page, language)
    messages = [AgentMessage.user(user_prompt)]
    chunks: list[str] = []

    print(f"\n--- [{page_index + 1}/{total_pages}] {page['title']} ---", flush=True)
    async for evt in run_agent_loop(
        cfg,
        messages,
        provider,
        tools,
        repo_path,
        qa_prompt_vars(repo_url, repo_type, repo_name, language),
    ):
        if isinstance(evt, TextDelta):
            chunks.append(evt.content)
            print_text_delta(evt.content)
        elif isinstance(evt, ToolCallStart):
            print_tool_call(evt.tool_name, evt.tool_args, indent="  ")
        elif isinstance(evt, ToolCallEnd):
            print_tool_end(evt, indent="  ")
        elif isinstance(evt, ErrorEvent):
            print(f"\n  [error] {evt.error}", file=sys.stderr)

    content = "".join(chunks)
    out_file = output_dir / f"{page_index + 1:02d}-{slugify(page['title'])}.md"
    out_file.write_text(content, encoding="utf-8")
    print(f"\n[saved] {out_file}", flush=True)


async def run_wiki_command(
    provider: UnifiedProvider,
    repo_path: str,
    repo_url: str,
    repo_type: str,
    repo_name: str,
    language: str,
    comprehensive: bool,
) -> None:
    """Orchestrate full two-phase wiki generation (/wiki command handler)."""
    output_dir = WIKI_OUTPUT_ROOT / repo_name
    output_dir.mkdir(parents=True, exist_ok=True)

    structure = await run_planner(
        provider, repo_path, repo_url, repo_type, repo_name, language, comprehensive
    )
    if structure is None:
        return

    # Clear stale pages from prior runs AFTER planner succeeds so a failed
    # planner run leaves the previous good output intact.
    for stale in output_dir.glob("*.md"):
        stale.unlink()

    pages = _flatten_pages_in_section_order(structure)
    for i, page in enumerate(pages):
        try:
            await run_writer_for_page(
                provider,
                repo_path,
                repo_url,
                repo_type,
                repo_name,
                language,
                page,
                i,
                len(pages),
                output_dir,
            )
        except Exception as exc:
            print(
                f"\n[page error] '{page.get('title', '?')}': {exc}",
                file=sys.stderr,
            )
            continue

    print(f"\n=== Wiki done: {len(pages)} pages -> {output_dir} ===")


# =============================================================================
# Section 6: REPL + main
# =============================================================================


async def repl(
    provider: UnifiedProvider,
    repo_url: str,
    repo_type: str,
    repo_name: str,
    repo_path: str,
    language: str,
    comprehensive: bool,
) -> None:
    """Interactive REPL: Q&A turns with accumulated history + /wiki command."""
    qa_cfg = get_agent_config("wiki")
    qa_tools = get_tools_for_agent(qa_cfg, repo_path)
    qa_vars = qa_prompt_vars(repo_url, repo_type, repo_name, language)
    history: list[AgentMessage] = []

    print(f"\nDeepWiki CLI ready. Repo: {repo_url}")
    print("Commands:  /wiki   /clear   /exit   (Ctrl+C also exits)\n")

    while True:
        try:
            raw = await asyncio.to_thread(input, "> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
        line = raw.strip()
        if not line:
            continue
        if line in ("/exit", "/quit"):
            return
        if line == "/clear":
            history.clear()
            print("[history cleared]")
            continue
        if line == "/wiki":
            await run_wiki_command(
                provider, repo_path, repo_url, repo_type, repo_name,
                language, comprehensive,
            )
            continue
        try:
            await run_qa_turn(
                line, history, provider, qa_cfg, qa_tools, repo_path, qa_vars
            )
        except Exception as exc:
            print(f"\n[error] {exc}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    provider_name, model = resolve_provider_model(args)
    repo_name, repo_path = compute_repo_meta(args.repo_url, args.repo_type)

    print(f"Provider : {provider_name}/{model}")
    print(f"Repo     : {repo_path}")
    print(f"Language : {_LANGUAGE_NAMES.get(args.language, args.language)}")
    print(
        f"Wiki mode: "
        f"{'comprehensive (15-20 pages)' if args.comprehensive else 'concise (8-10 pages)'}"
    )
    # print()
    # print("Building FAISS index (first run may take a while)...")
    # DatabaseManager().prepare_retriever(args.repo_url, args.repo_type, args.token)
    DatabaseManager()._create_repo(repo_url_or_path=args.repo_url, repo_type=args.repo_type, access_token=args.token)

    provider = UnifiedProvider(provider_name, model)
    asyncio.run(
        repl(
            provider,
            args.repo_url,
            args.repo_type,
            repo_name,
            repo_path,
            args.language,
            args.comprehensive,
        )
    )


if __name__ == "__main__":
    main()
