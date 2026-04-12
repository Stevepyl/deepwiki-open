import os
import argparse

from adalflow.core.db import LocalDB
from adalflow.utils import get_adalflow_default_root_path


def load_db(repo_name: str) -> LocalDB:
    """
    加载指定 repo_name 对应的 LocalDB.

    本地路径仓库时，repo_name 一般就是目录名，例如：
    deepwiki-open -> ~/.adalflow/databases/deepwiki-open.pkl
    """
    root = get_adalflow_default_root_path()
    db_path = os.path.join(root, "databases", f"{repo_name}.pkl")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    print(f"Loading DB from: {db_path}")
    return LocalDB.load_state(db_path)


def print_chunks(repo_name: str, limit: int) -> None:
    db = load_db(repo_name)
    docs = db.get_transformed_data(key="split_and_embed") or []

    total = len(docs)
    print(f"Total chunks in DB: {total}")

    if total == 0:
        return

    limit = max(1, min(limit, total))

    for i, doc in enumerate(docs[:limit]):
        meta = getattr(doc, "meta_data", {}) or {}
        file_path = meta.get("file_path")
        title = meta.get("title")
        is_code = meta.get("is_code")
        token_count = meta.get("token_count")

        # best-effort 计算向量维度
        vec = getattr(doc, "vector", None)
        vec_dim = None
        if vec is not None:
            try:
                if hasattr(vec, "shape") and getattr(vec, "shape", None) is not None:
                    # numpy / torch-like
                    if len(vec.shape) == 1:
                        vec_dim = int(vec.shape[0])
                    else:
                        vec_dim = int(vec.shape[-1])
                elif hasattr(vec, "__len__"):
                    vec_dim = len(vec)
            except Exception:
                vec_dim = "unknown"

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

#repo name:AsyncFuncAI_deepwiki-open,xuanheya_deepwiki-open,xuanheya_AGENT,pjdaye_AGENT
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect embedded chunks stored in LocalDB for a repo."
    )
    parser.add_argument(
        "--repo-name",
        default="xuanheya_AGENT",
        help="Repository name used for DB filename (default: deepwiki-open)",
    )
    parser.add_argument(
        "-n",
        "--num-chunks",
        type=int,
        default=10,
        help="Number of chunks to print (default: 10)",
    )

    args = parser.parse_args()
    print_chunks(repo_name=args.repo_name, limit=args.num_chunks)


if __name__ == "__main__":
    main()
