import argparse
import os
from collections import Counter


def load_db(repo_name: str):
    """
    Load the LocalDB file for the given repo name.
    """
    from adalflow.core.db import LocalDB
    from adalflow.utils import get_adalflow_default_root_path

    root = get_adalflow_default_root_path()
    db_path = os.path.join(root, "databases", f"{repo_name}.pkl")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    print(f"Loading DB from: {db_path}")
    return LocalDB.load_state(db_path)


def _vector_dim(doc) -> int | str | None:
    vec = getattr(doc, "vector", None)
    if vec is None:
        return None

    try:
        if hasattr(vec, "shape") and getattr(vec, "shape", None) is not None:
            if len(vec.shape) == 1:
                return int(vec.shape[0])
            return int(vec.shape[-1])
        if hasattr(vec, "__len__"):
            return int(len(vec))
    except Exception:
        return "unknown"

    return None


def print_summary(docs: list, top_files: int) -> None:
    total = len(docs)
    print(f"Total chunks in DB: {total}")

    if total == 0:
        return

    file_counter: Counter[str] = Counter()
    type_counter: Counter[str] = Counter()
    files_with_ast_metadata = 0
    vector_dims = Counter()

    for doc in docs:
        meta = getattr(doc, "meta_data", {}) or {}
        file_path = meta.get("file_path") or "<unknown>"
        file_counter[file_path] += 1

        doc_type = meta.get("type") or "<unknown>"
        type_counter[doc_type] += 1

        if meta.get("ast_chunk_count") is not None:
            files_with_ast_metadata += 1

        vec_dim = _vector_dim(doc)
        if vec_dim is not None:
            vector_dims[str(vec_dim)] += 1

    print(f"Unique files      : {len(file_counter)}")
    print(f"Docs with AST meta: {files_with_ast_metadata}")

    print("\nChunks by file type:")
    for doc_type, count in type_counter.most_common():
        print(f"  {doc_type:>10}: {count}")

    print(f"\nTop {min(top_files, len(file_counter))} files by chunk count:")
    for file_path, count in file_counter.most_common(top_files):
        print(f"  {count:>4}  {file_path}")

    if vector_dims:
        print("\nVector dimensions:")
        for dim, count in vector_dims.most_common():
            print(f"  {dim:>10}: {count}")


def print_chunk_details(docs: list, limit: int) -> None:
    total = len(docs)
    if total == 0 or limit <= 0:
        return

    limit = min(limit, total)

    for i, doc in enumerate(docs[:limit], start=1):
        meta = getattr(doc, "meta_data", {}) or {}
        file_path = meta.get("file_path")
        title = meta.get("title")
        is_code = meta.get("is_code")
        token_count = meta.get("token_count")
        vec_dim = _vector_dim(doc)

        text_preview = (getattr(doc, "text", "") or "").strip().replace("\r\n", "\n")
        if len(text_preview) > 400:
            text_preview = text_preview[:400] + "..."

        print("\n" + "=" * 80)
        print(f"Chunk #{i}")
        print(f"  meta_data   : {meta}")
        print(f"  file_path   : {file_path}")
        print(f"  title       : {title}")
        print(f"  is_code     : {is_code}")
        print(f"  token_count : {token_count}")
        print(f"  vector_dim  : {vec_dim}")
        print("  text preview:")
        print("  " + "-" * 76)
        for line in text_preview.split("\n"):
            print("  " + line)


def inspect_chunks(repo_name: str, detail_limit: int, top_files: int) -> None:
    db = load_db(repo_name)
    docs = db.get_transformed_data(key="split_and_embed") or []

    print_summary(docs, top_files=top_files)
    print_chunk_details(docs, limit=detail_limit)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect embedded chunks stored in LocalDB for a repo."
    )
    parser.add_argument(
        "--repo-name",
        default="xuanheya_AGENT",
        help="Repository name used for DB filename.",
    )
    parser.add_argument(
        "--top-files",
        type=int,
        default=20,
        help="How many files to show in the chunk count ranking (default: 20).",
    )
    parser.add_argument(
        "-n",
        "--num-chunks",
        type=int,
        default=0,
        help="Number of chunk details to print after the summary (default: 0).",
    )

    args = parser.parse_args()
    inspect_chunks(
        repo_name=args.repo_name,
        detail_limit=max(0, args.num_chunks),
        top_files=max(1, args.top_files),
    )


if __name__ == "__main__":
    main()
