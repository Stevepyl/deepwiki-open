# HANDBOOK

## 这个项目是如何把代码变成结构化 Wiki 的
Insight
- 代码不会直接原样喂给模型，而是先被整理成“对话历史 + 当前文件 + 检索上下文 + 用户问题”四层上下文。
- `api/websocket_wiki.py` 负责请求编排和流式输出，`api/prompts.py` 负责定义模型应该如何以结构化 Markdown 形式写出结果。
- 所谓“生成 Wiki”，本质上是 RAG 检索出的代码证据经过 prompt 约束后，被模型整理成章节化说明。

可以按 6 个阶段理解：
1. 接收前端请求并结构化参数
    入口在 `api/websocket_wiki.py` 的 `handle_websocket_chat`。
    WebSocket 收到 JSON 后，会解析为 `ChatCompletionRequest`，里面包含：
   - `repo_url`
   - `messages`
   - `filePath`
   - `provider`
   - `model`
   - `language`
   - included / excluded 过滤规则
    这一步的意义是先把“分析哪个仓库、问什么问题、是否只关注某个文件、用哪个模型回答”全部结构化。

2. 构建 RAG 检索器，让仓库代码变成可召回的上下文
    `api/websocket_wiki.py` 会创建 `RAG(provider=request.provider, model=request.model)`，然后调用 `prepare_retriever(...)`。
    这一层虽然具体逻辑在 `api/rag.py` 和 `api/data_pipeline.py`，但从接口可以看出它会：
   - 根据仓库地址准备本地检索数据
   - 应用 included / excluded 文件过滤规则
   - 依赖 embedding 结果建立检索能力
    这说明系统的关键不是把整个仓库一次塞给模型，而是先把代码转成可检索的知识库。

3. 把历史对话整理成连续语境
    在 `api/websocket_wiki.py` 中，旧的 user / assistant 消息会被写入 `request_rag.memory`。
    后面再格式化成：
   - `<turn><user>...</user><assistant>...</assistant></turn>`
    这一步让模型知道前面已经讨论过什么，避免每轮都从零开始解释仓库。

4. 执行 RAG 检索，并把零散 chunk 重新组织成“按文件聚合的证据包”
    如果输入没有过大，系统会调用 `request_rag(rag_query, language=request.language)` 检索相关文档。
    有两个关键点：
   - 如果用户指定了 `filePath`，检索 query 会改写成围绕该文件的上下文查询
   - 检索回来的文档不会直接原样拼接，而是按 `file_path` 分组
    之后上下文会被组织成类似：
   - `## File Path: xxx`
   - 该文件下的多个相关片段内容
    这样做的好处是，模型看到的是“面向文件的知识包”，而不是随机散落的文本块，更容易写出模块说明、职责划分和章节结构。

5. 追加目标文件原文，补足局部精确上下文
    如果请求里带了 `filePath`，系统还会通过 `get_file_content(...)` 把目标文件完整内容取出来，并放进：
   - `<currentFileContent path="...">...</currentFileContent>`
    这样模型既能看见：
   - RAG 检索得到的相关片段
   - 当前目标文件的完整原文
    于是它既能做宏观总结，也能引用具体实现细节。

6. 使用 prompt 模板，把上下文压成“结构化写作任务”
    最后 `api/websocket_wiki.py` 会把以下内容拼成一个统一 prompt：
   - system prompt
   - conversation history
   - current file content
   - retrieved context
   - user query
    `api/prompts.py` 则定义了这些 prompt 的写法规则。

    其中 `RAG_SYSTEM_PROMPT` 规定：
   - 自动跟随用户语言回答
   - 强制使用 Markdown 组织内容
   - 文件路径使用 `inline code`
   - 不要额外包 ```markdown fence

    `RAG_TEMPLATE` 则把信息明确分层：
   - `<START_OF_SYS_PROMPT>`
   - `<START_OF_CONVERSATION_HISTORY>`
   - `<START_OF_CONTEXT>`
   - `<START_OF_USER_PROMPT>`
    这种分层标签非常重要，因为它告诉模型：哪些是规则，哪些是历史，哪些是代码证据，哪些才是当前要回答的问题。

## Deep Research 模式如何进一步强化结构化输出
Insight
- Deep Research 不是一次性给结论，而是把结构化文档拆成“计划、增量研究、最终结论”几个阶段生成。
- 这让模型更像在逐步撰写一份研究型 Wiki，而不是一次性吐出一段回答。
- 阶段化 prompt 是保证输出有章节感和连续性的关键。

`api/websocket_wiki.py` 会检查消息里是否带有 `[DEEP RESEARCH]` 标记。
如果有，就会根据轮次切换不同 prompt：
- 第一轮：输出 `## Research Plan`，给出研究方向和初步发现
- 中间轮：输出 `## Research Update N`，补充新的研究结果，避免重复
- 最终轮：输出 `## Final Conclusion`，综合前面所有发现

这些模板定义在 `api/prompts.py`：
- `DEEP_RESEARCH_FIRST_ITERATION_PROMPT`
- `DEEP_RESEARCH_INTERMEDIATE_ITERATION_PROMPT`
- `DEEP_RESEARCH_FINAL_ITERATION_PROMPT`

这种设计让系统能在多轮对话中逐步收敛到一份更完整、更有层次的结构化说明。

## 为什么最终结果看起来像 Wiki
Insight
- 这里的“Wiki 感”不是来自前端样式，而是来自后端已经把生成任务定义成了 Markdown 文档写作。
- 模型输出的每个片段都在强约束下生成，因此天然更接近章节化文档，而不是普通聊天回复。
- WebSocket 流式返回再配合前端 Markdown 渲染，就形成了“边生成边长出页面”的体验。

原因主要有四个：
1. 上下文本身是结构化的
   - 对话历史、当前文件、检索片段、用户问题都被标签化
2. prompt 明确要求 Markdown 输出
   - 标题、列表、表格、代码块都被鼓励使用
3. Deep Research prompt 强制阶段性结构
   - `Research Plan`、`Research Update`、`Final Conclusion` 都天然适合 Wiki / 报告式表达
4. 响应是流式发送的
   - `api/websocket_wiki.py` 会把模型生成的 chunk 持续 `send_text(...)` 给前端
   - 前端可以边接收边渲染 Markdown，因此用户看到的是一个逐步成形的 Wiki 页面

## 一句话总结
DeepWiki-Open 并不是通过编译器级静态分析把代码直接转换成 Wiki。
它的真实路径是：先把仓库代码切块和向量化，检索出与问题最相关的代码证据，再把这些证据和文件原文、历史对话一起组织成分层 prompt，最后用 LLM 按 Markdown / Research 模板生成结构化说明。

因此，它擅长生成“面向人阅读的仓库知识文档”，而不是严格形式化的程序分析结果。

## 从 repo URL 到 Wiki 页面渲染的完整调用链
Insight
- 这条链路不是单点函数调用，而是前端页面、Next.js 代理、FastAPI WebSocket、RAG/数据管道、流式渲染协同工作的结果。
- `src/app/[owner]/[repo]/page.tsx` 是前端消费流的入口，`api/api.py` 是后端入口，`api/websocket_wiki.py` 是真正串起生成流程的编排层。
- 页面之所以能“边生成边显示”，不是因为前端轮询，而是因为后端持续通过 WebSocket 推送 Markdown chunk。

可以按 8 个阶段理解：
1. 首页收集 repo URL 并进入目标路由
    前端起点是 `src/app/page.tsx`。
    这里负责：
   - 接收用户输入的 GitHub / GitLab / Bitbucket 仓库地址
   - 选择模型或相关参数
   - 解析出 `owner/repo`
   - 导航到 `src/app/[owner]/[repo]/page.tsx`
    这一层更像“任务发起器”，负责把“分析哪个仓库”变成明确的页面状态。

2. Wiki 页面初始化并选择缓存或实时生成路径
    到达 `src/app/[owner]/[repo]/page.tsx` 后，页面会：
   - 根据路由参数确认当前仓库
   - 初始化页面状态
   - 尝试读取已有 wiki 缓存
   - 如果没有缓存或需要重建，则建立 WebSocket 连接
    因此这里其实有两条路径：
   - 缓存命中：直接读取已有结果并渲染
   - 实时生成：连接后端并逐步接收生成内容

3. Next.js 通过 rewrite/proxy 把前端请求转发给 Python 后端
    `next.config.ts` 中定义了若干 rewrite 规则，例如：
   - `/api/wiki_cache/*`
   - `/export/wiki/*`
   - `/api/auth/*`
    这意味着前端表面上访问的是本站路径，实际上会被透明转发到 Python backend。
    这样做的作用是：
   - 统一前后端入口
   - 降低跨域处理复杂度
   - 让前端组件不需要硬编码后端地址

4. FastAPI 在 `api/api.py` 中接住 REST 和 WebSocket 请求
    `api/api.py` 是后端总入口。
    它负责：
   - 注册 REST API
   - 注册 WebSocket endpoint
   - 配置 CORS
   - 把请求分发到缓存读取、鉴权、聊天、Wiki 生成等逻辑
    在这条调用链中，它相当于一个路由枢纽：
   - 短请求走 REST
   - 长生命周期的 wiki 生成走 WebSocket

5. `api/websocket_wiki.py` 编排整个 Wiki 生成流程
    这是最关键的 orchestrator。
    它负责：
   - 接收 repo URL、消息、模型配置等请求参数
   - 初始化 RAG / 检索器
   - 准备 prompt 所需上下文
   - 按章节或按阶段生成 wiki 内容
   - 持续把结果通过 WebSocket 发回前端
    因此它不是单纯“调用一下模型”，而是在协调：
   - 数据准备
   - 上下文组织
   - 模型生成
   - 流式发送

6. `api/data_pipeline.py` 把仓库代码加工成可检索知识库
    在 wiki 真正生成前，系统需要先把 repo 变成适合检索的结构化材料。
    `api/data_pipeline.py` 负责的通常包括：
   - clone 仓库到 `~/.adalflow/repos/`
   - 根据 `api/config/repo.json` 过滤文件
   - 把代码切成 chunk（项目当前策略是 350-word chunk + 100-word overlap）
   - 调用 embedding provider 生成向量
   - 把结果写入 `~/.adalflow/databases/` 下的 FAISS 索引
    这一步的核心意义是：
   - 不是把整个仓库一次性喂给模型
   - 而是先建立一个“可召回的代码知识库”

7. `api/prompts.py` 和 provider clients 决定“如何写成 Wiki”
    有了代码证据之后，还需要两层能力：
   - `api/prompts.py`：规定模型应该如何组织输出
   - provider clients：真正去调用 embedding / chat / stream 接口
    `api/prompts.py` 会定义：
   - 输出应使用 Markdown
   - 如何组织章节、标题、列表、代码引用
   - Deep Research 模式下如何分阶段输出 `Research Plan`、`Research Update`、`Final Conclusion`
    provider clients（如 `openai_client.py`、`openrouter_client.py`、`google_embedder_client.py`、`ollama_patch.py`）则负责：
   - 把 chunk 转成 embedding
   - 基于 prompt + context 调用模型生成文本
   - 以流式方式返回 token / chunk

8. 前端接收 WebSocket 增量消息并渲染成页面
    后端生成出的内容会被 `api/websocket_wiki.py` 持续 `send_text(...)` 给前端。
    `src/app/[owner]/[repo]/page.tsx` 负责接收这些增量消息，并更新页面状态。
    随后不同类型的内容交给不同组件渲染：
   - `src/components/Markdown.tsx`：渲染 Markdown 与代码高亮
   - `src/components/Mermaid.tsx`：渲染 Mermaid 图表
   - 其他组件：负责树视图、交互区、补充 UI
    因为是“收到一点就更新一点”，所以用户看到的效果就是：
   - 后端在分析和生成
   - 前端页面同步逐步长出来

## 用一张简化链路图理解
```text
用户输入 repo URL
→ src/app/page.tsx 收集参数并跳转
→ src/app/[owner]/[repo]/page.tsx 初始化页面
→ next.config.ts rewrite/proxy 转发请求
→ api/api.py 接住 REST / WebSocket
→ api/websocket_wiki.py 编排 Wiki 生成
→ api/data_pipeline.py clone / filter / chunk / embedding / FAISS
→ api/prompts.py 组织提示词
→ provider clients 调用 embedding + LLM stream
→ api/websocket_wiki.py 持续发送 Markdown chunk
→ src/app/[owner]/[repo]/page.tsx 更新 state
→ Markdown.tsx / Mermaid.tsx 渲染
→ 用户看到逐步成形的 Wiki 页面
```

## 为什么这条链路重要
Insight
- 这条链路解释的是“系统怎么工作”，而不只是“模型看到了什么”。
- 如果生成结果不对，通常可以沿着这条链路排查：文件过滤是否过严、检索是否召回正确、prompt 是否约束到位、前端是否正确消费流。
- 这也是理解 DeepWiki-Open 架构最实用的入口，因为它同时覆盖了前端、后端、检索和渲染。

从工程视角看，DeepWiki-Open 的核心价值链可以概括为：
- `repo URL → local repo → chunks → embeddings → retrieved context → prompted generation → streamed UI`

这说明它本质上是一个“面向仓库知识生成的流式 RAG 系统”，而不是传统意义上的静态分析器。
