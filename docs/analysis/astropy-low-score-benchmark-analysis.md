---
number: DOC-009
name: Astropy 低分 Benchmark 分析
description: 通过比较评测分数、原始输出和 ground truth reference，分析 Astropy benchmark 中最低的非异常答案。
update_at: 2026-05-05
category: analysis
language: zh-CN
audience: developers-and-agents
---

# Astropy 低分 Benchmark 分析

## 范围

本文分析 `benchmark/output/eval/astropy.jsonl` 和
`benchmark/output/raw_results/astropy.jsonl`。

eval 文件包含 48 条带评分的 JSON 记录，末尾还有 7 行以注释形式记录的聚合均值。总分计算方式为：

```text
correctness + completeness + clarity + relevance + reasoning
```

绝对最低的记录总分为 `5`、`5` 和 `9`。由于请求明确排除了类似 “total score is just 10 etc.” 的案例，这些记录被视为异常值。因此，最低的五条非异常记录从总分 `47` 开始。

## 最低的五个非异常案例

| 排名 | Eval 行 | Raw 行 | Ground truth 行 | 总分 | 分数拆解 |
|---:|---:|---:|---:|---:|---|
| 1 | 46 | 43 | 42 | 47 | correctness 5, completeness 5, clarity 15, relevance 12, reasoning 10 |
| 2 | 37 | 40 | 39 | 56 | correctness 4, completeness 8, clarity 16, relevance 12, reasoning 16 |
| 3 | 48 | 49 | 48 | 57 | correctness 5, completeness 5, clarity 19, relevance 18, reasoning 10 |
| 4 | 28 | 24 | 23 | 58 | correctness 12, completeness 10, clarity 14, relevance 10, reasoning 12 |
| 5 | 33 | 36 | 35 | 64 | correctness 2, completeness 8, clarity 18, relevance 20, reasoning 16 |

## 失败分析

### 1. FITS keyword normalization 的位置

问题：

> 在代码库中，那个会在 FITS keyword strings 被用作 header 的 keyword-to-index mapping dictionary 的 key 之前，将它们转换为大写的 classmethod 位于哪里？

Ground truth 将 `astropy/io/fits/header.py` 中的 `Header._fromcards` 识别为目标 classmethod。它调用 `Card.normalize_keyword`，随后把归一化后的值用于 `_keyword_indices`。

原始答案则把 `astropy/io/fits/card.py` 中的 `Card.normalize_keyword()` 命名为目标 classmethod。这个答案部分相关，因为它确实执行了大写归一化，但它漏掉了问题中的约束：“before they are used as keys in the header's keyword-to-index mapping dictionary”。这个 mapping 的使用位置发生在 `Header._fromcards`，而不是 `Card.normalize_keyword` 内部。

低分原因：答案停在了执行字符串转换的 helper method，没有沿着数据流继续追踪到 header mapping 的构建位置。

### 2. SIP distortion 的控制流

问题：

> 在 Simple Imaging Polynomial distortion correction method 的控制流中，输入坐标的转换是在什么位置先经过 intermediate shifted values，然后再应用 polynomial distortion？

Ground truth 指向 C 实现：`astropy/wcs/src/sip.c` 中的 `sip_compute()`。在该函数中，输入坐标会先减去 `crpix` 形成 shifted values，然后再进入多项式计算。它还给出了运行时路径：`WCS.pix2foc()`、`_pix2foc`、`pipeline_pix2foc()`、`sip_pix2deltas()`，最后到 `sip_compute()`。

原始答案指向 Python modeling 实现：`astropy/modeling/polynomial.py` 中的 `SIP.evaluate()`。这段代码具有相似的 “shift 后再多项式计算” 结构，但它不是 benchmark reference 所询问的 WCS SIP distortion correction 控制流。

低分原因：词法搜索找到了名为 `SIP` 的类，答案随即锁定 Python modeling 抽象，漏掉了问题期望的更底层 WCS C 路径。

### 3. Quantity numpy override helper 的 imports

问题：

> 在提供 Quantity-specific numpy function override implementations 的 helper module 中，如果检查 array dimensions 和 structure 的 numpy functions 被从 “numpy functions 到 custom implementations” 的 mapping dictionary 中移除，会影响哪些 imported modules？

Ground truth 将相关文件定位为 `astropy/units/quantity_helper/function_helpers.py`，并将受影响的 imported modules 识别为 `numpy`，以及条件导入的 `numpy.core` / `numpy._core`。相关例子集中在维度和结构函数上，例如 `shape`、`size`、`ndim`、`expand_dims`、`squeeze`、`reshape`、`transpose` 和 `broadcast_to`。

原始答案选择了 `DISPATCHED_FUNCTIONS`，并聚焦于 `np.array_equal`、`np.array_equiv` 和 `np.block`。benchmark reference 期望的则是 `SUBCLASS_SAFE_FUNCTIONS` 中的维度/结构函数，以及它们对 numpy imports 的依赖。

低分原因：答案错误识别了相关 mapping。它把 “custom implementations” 理解为 dispatched helper functions，而 reference question 指向的是 subclass-safe numpy functions；移除这些函数会改变 Quantity 对象与 numpy 结构检查函数的交互方式。

### 4. 来自异构列来源的 Table initialization

问题：

> 验证 table initialization from heterogeneous column sources 的 test class，在用 mixed column inputs 创建 tables 时，是如何强制区分 column name/type resolution logic 与 parent table reference assignment 的？

Ground truth 聚焦于 `TestInitFromColsList` 以及实现中的职责拆分：`_convert_data_to_col` 负责 name 和 dtype resolution，而 `_set_col_parent_table_and_mask` 稍后由 `_make_table_from_cols` 调用，用于分配 parent table references。

原始答案找到了 `TestInitFromColsList` 和相关测试，但把讨论扩展到了多个 test classes。它强调了 names、dtypes 以及 copy/reference behavior，却没有清楚地把测试断言连接到 column conversion 与 parent assignment 之间的实现边界。

低分原因：局部检索正确，但综合归纳发散。答案描述了附近的测试，而不是明确说出具体的 separation contract，以及承载该契约的两个实现函数。

### 5. Logarithmic quantity `_unit_class` 的继承层级

问题：

> 对 logarithmic quantities 来说，指定 function unit class 的 class attribute 是在继承层级的哪一层控制从 base class 继承来的 instantiation 和 conversion behavior？

Ground truth 认为控制层级是 `LogQuantity`：`FunctionQuantity` 定义 `_unit_class = None` 并实现被继承的行为，而 `LogQuantity` 设置 `_unit_class = LogUnit`。具体子类随后用 `DexUnit`、`DecibelUnit` 和 `MagUnit` 覆盖该属性。

原始答案列出了这个继承层级，但结论是 `_unit_class` 定义在 `FunctionQuantity` 层级。这混淆了 base placeholder declaration 与 logarithmic quantities 实际建立 function unit behavior 的层级。

低分原因：答案找到了正确文件和继承层级，但在 “method is implemented in the base class” 与 “logarithmic behavior is controlled at the subclass level” 之间做出了错误的语义区分。

## 跨案例模式

这些低分并不主要来自答案不可读。其 clarity 通常处于中等到较高水平。失败主要是语义定位错误：

- 答案找到了 helper，但漏掉了满足 benchmark 条件的 caller。
- 答案匹配到名称相似的 abstraction，但没有匹配到期望的 runtime path。
- 当同一文件中存在多个看似合理的 mappings 时，答案选择了错误的 internal mapping。
- 答案总结了相邻测试，却没有把它们绑定到问题询问的 implementation boundary。
- 答案混淆了 “base class defines the mechanism” 与 “subclass level controls the concrete behavior”。

对于这个 benchmark，模型在最终作答前需要更强的数据流和调用路径验证。raw outputs 显示，简单文本搜索通常已经找到了相关文件，但最终答案经常停在 ground truth 所使用的精确代码位置或语义边界之前一步。

## “停在精确代码位置之前一步”的含义

“停在精确代码位置之前一步”指的是：模型已经找到了相关代码，但它停在了看起来最直接相关的位置，没有继续追踪到 ground truth 真正要求的使用点、调用点、控制流节点或语义边界。它不是完全偏离主题，而是没有完成最后一步验证。

在 FITS keyword normalization 案例中，原始答案选择 `Card.normalize_keyword()`。这个函数确实负责把 keyword 归一化为大写，因此它是相关 helper。但问题问的是 keyword 在被用作 header 的 keyword-to-index mapping key 之前，相关 classmethod 位于哪里。满足这个条件的位置是 `Header._fromcards()`：它调用 `Card.normalize_keyword()`，然后把 normalized keyword 放入 `_keyword_indices`。也就是说，正确链路是：

```text
Card.normalize_keyword()
    -> 被 Header._fromcards() 调用
    -> normalized keyword 被用于 _keyword_indices
```

原始答案停在第一步，没有继续走到 mapping construction site。

在 SIP distortion 案例中，原始答案找到了 `astropy/modeling/polynomial.py::SIP.evaluate()`。这段代码确实有类似的 “shift coordinates 后执行 polynomial calculation” 结构。但 benchmark reference 问的是 WCS SIP distortion correction 的实际运行控制流，路径是：

```text
WCS.pix2foc()
    -> _pix2foc
    -> pipeline_pix2foc()
    -> sip_pix2deltas()
    -> sip_compute()
```

精确位置是 `astropy/wcs/src/sip.c::sip_compute()`。原始答案匹配到了名称和代码形状都很像的 `SIP` abstraction，但没有验证它是否处在 reference 期望的 WCS runtime path 中。

在 Table initialization 案例中，原始答案找到了 `TestInitFromColsList`，也讨论了 names、dtypes 和 copy/reference behavior。这说明检索范围基本正确。但 ground truth 要的是测试如何对应到实现职责拆分：`_convert_data_to_col()` 负责 column name 和 dtype resolution，`_make_table_from_cols()` 后续调用 `_set_col_parent_table_and_mask()` 来处理 parent table reference assignment。原始答案停在相关测试和相邻行为总结，没有把测试断言落实到两个具体实现函数之间的边界。

因此，这类错误的核心不是 “找不到相关文件”，而是 “找到相关文件后，没有继续验证最后一个代码位置、调用路径或语义边界”。对 benchmark 答案来说，最后一步通常决定 correctness：helper、相似 abstraction、附近测试都只能算候选证据，必须继续追踪到问题条件真正被满足的代码点。
