---
number: RISK-003
name: Repository Cache Key Collision
description: Records the cache key collision risk caused by owner-repo naming across hosts, forks, and private repositories.
update_at: 2026-05-04
category: risk-record
language: zh-CN
status: open
---

# 仓库缓存键冲突风险

## 背景

DeepWiki-Open 将克隆的仓库、向量数据库和 wiki 缓存存储在 `~/.adalflow/` 目录下，使用 `{owner}_{repo}` 作为所有子目录的缓存键。这个命名方案在最初针对单一 GitHub 场景设计时是合理的，但随着系统支持 GitHub Enterprise、GitLab、Bitbucket 和私有实例后，它引入了一个系统性的缓存键冲突风险。

Codex adversarial review（2026-04-19）将此问题标记为 [high] 安全风险，指出 subtask 12 (`api/agent/wiki_generator.py:95-107`) 继承了相同的冲突缺陷。

---

## 影响范围

该风险影响以下四个子系统，所有子系统使用相同或类似的命名方案：

| 子系统 | 路径格式 | 相关代码 |
|--------|----------|----------|
| 克隆仓库 | `~/.adalflow/repos/{owner}_{repo}/` | `api/data_pipeline.py:804` |
| 向量数据库 | `~/.adalflow/databases/{owner}_{repo}.pkl` | `api/data_pipeline.py:813` |
| Wiki 缓存 | `~/.adalflow/wikicache/deepwiki_cache_{type}_{owner}_{repo}_{lang}_{mode}` | `api/api.py:92` |
| Agent wiki 路径（subtask 12）| `~/.adalflow/repos/{owner}_{repo}/` | `api/agent/wiki_generator.py:97-104` |

---

## 风险场景

### 场景 1：跨 Host 冲突

`https://github.com/my-org/my-repo` 和 `https://my-company.github.internal/my-org/my-repo` 具有相同的 `owner` 和 `repo` 名称，但指向完全不同的代码库。两者都会被克隆到 `~/.adalflow/repos/my-org_my-repo/`。

**后果**：第一次克隆成功后，第二次请求触发 `download_repo` 时，`data_pipeline.py:807` 的 `os.path.exists(save_repo_dir) and os.listdir(save_repo_dir)` 检查为 truthy，**跳过克隆**，直接使用第一次的结果。第二次请求将对错误的代码库生成 wiki。

### 场景 2：私有/公有 slug 重叠

用户 A 通过私有 PAT 克隆了 `https://github.com/org/private-repo`（内含 proprietary 代码）。用户 B 随后请求分析 `https://github.com/org/private-repo` 但不提供 token（公有访问失败或返回空内容）。然而，由于目录已存在且非空，用户 B 的请求会**复用用户 A 的克隆**。

**后果**：用户 B 的 wiki 和 agent 工具查询能读取到本应私有的代码。这是一个**信息泄露漏洞**。

### 场景 3：同名 Fork 与 Original

`https://github.com/alice/awesome-lib`（fork）与 `https://github.com/original/awesome-lib`（original）具有相同的 `repo` 名但不同的 `owner`。

**当前行为**：`data_pipeline.py:770-772` 使用 `owner_repo` 格式，`owner` 不同所以路径不同（`alice_awesome-lib` vs `original_awesome-lib`）——**这个场景实际上是安全的**，因为两个 owner 不同。

但如果 `owner` 也相同（如企业 GitHub 与 GitHub.com 上都有 `company/backend`），则会冲突。

### 场景 4：并发克隆竞争（已知但非新问题）

两个并发请求同一仓库时，第一个请求的 `git clone` 正在进行（目录已创建但内容不完整），第二个请求的 `os.path.exists + os.listdir` 判断为 truthy，跳过等待，使用不完整的仓库副本。

---

## 根本原因

`api/data_pipeline.py:762-775` 的 `_extract_repo_name_from_url` 方法：

```python
def _extract_repo_name_from_url(self, repo_url_or_path: str, repo_type: str) -> str:
    url_parts = repo_url_or_path.rstrip('/').split('/')
    if repo_type in ["github", "gitlab", "bitbucket"] and len(url_parts) >= 5:
        owner = url_parts[-2]
        repo = url_parts[-1].replace(".git", "")
        repo_name = f"{owner}_{repo}"    # ← 丢失了 host 信息
    else:
        repo_name = url_parts[-1].replace(".git", "")
    return repo_name
```

关键问题：`scheme://host` 被丢弃，仅保留路径的最后两个段。`https://github.com/org/repo` 和 `https://mycompany.github.internal/org/repo` 生成相同的 `org_repo`。

---

## 为何不在 Subtask 12 修复

1. **先存在的系统性问题**：该缺陷在所有四个子系统中都存在，subtask 12 只是继承了既有行为。单独修复 `wiki_generator.py` 中的路径计算而不修复 `data_pipeline.py`，会导致 agent wiki 路径使用与 RAG 路径**不同**的克隆目录，需要两次克隆同一仓库。
2. **迁移成本高**：修改 key 方案意味着所有现有用户的 `~/.adalflow/repos/`、`databases/`、`wikicache/` 目录都需要重命名或迁移。需要专门的 migration script 和版本化迁移策略。
3. **需要跨系统协调**：wikicache 键在 `api/api.py:92` 中独立计算，需要与 data_pipeline 的键方案同步修改。
4. **范围管控**：Codex 在 review 时明确指出这是预存在的架构缺陷，不是 subtask 12 新引入的。在 subtask 12 的 PR 中修复会使 review scope 扩大 3 倍。

---

## 建议重新设计方案

### 新缓存键格式

```
key = base64url_nopad(sha256(normalized_url))
```

**normalized_url 规则**：
- 小写
- 去尾斜杠
- 去 `.git` 后缀
- 保留完整 `scheme://host/owner/repo`

示例：
```
https://github.com/org/repo     →  sha256("https://github.com/org/repo")  →  "aB3x..."
https://myco.github.com/org/repo →  sha256("https://myco.github.com/org/repo") → "kP7y..."  # 不同！
```

### 元数据文件

每个克隆目录下保存 `.deepwiki-meta.json`：

```json
{
  "url": "https://github.com/org/repo",
  "normalized_url": "https://github.com/org/repo",
  "cloned_at": 1713484800,
  "last_verified": 1713484800
}
```

### 克隆前验证

在使用已有目录之前，验证 remote origin 与请求 URL 一致：

```python
result = subprocess.run(
    ["git", "remote", "get-url", "origin"],
    cwd=save_repo_dir, capture_output=True, text=True
)
if normalize_url(result.stdout.strip()) != normalize_url(requested_url):
    # URL 不匹配 → 重命名旧目录（保留审计）并重新克隆
    shutil.move(save_repo_dir, f"{save_repo_dir}.stale.{int(time.time())}")
    download_repo(...)
```

### 向后兼容迁移

```bash
# migration script 伪代码
for old_dir in ~/.adalflow/repos/*/; do
    git -C "$old_dir" remote get-url origin > /tmp/origin.txt
    new_key=$(python -c "import hashlib,base64; ...")
    mv "$old_dir" ~/.adalflow/repos/"$new_key"/
    # 同步重命名对应的 .pkl 和 wikicache 条目
done
```

过渡期：提供 `deepwiki cleanup-cache --migrate` 命令，允许用户手动触发迁移而非强制在升级时执行。

---

## 优先级建议

| 优先级 | 条件 |
|--------|------|
| 高（尽快） | 若部署为公共服务（多用户，接受任意 URL 输入） |
| 中（下一个大版本） | 若为内部/单用户部署，但打算支持 GitHub Enterprise |
| 低（技术债积压） | 若仅在 GitHub.com 单一 host 上使用 |

---

## 相关文件

| 文件 | 问题位置 |
|------|----------|
| `api/data_pipeline.py:762-775` | `_extract_repo_name_from_url` — 根本原因 |
| `api/data_pipeline.py:804-811` | `_create_repo` — `os.path.exists` 短路逻辑 |
| `api/api.py:92` | wikicache key 计算（独立但同类问题） |
| `api/agent/wiki_generator.py:97-104` | `_compute_repo_name` — 继承的缺陷 |

## 决策记录

- **2026-04-19**：Codex adversarial review 将此问题标记为 [high]，确认为 subtask 12 前已存在的架构缺陷。本 handbook 记录风险、根本原因和未来重新设计方案，不修改代码。
