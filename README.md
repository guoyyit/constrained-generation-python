# constrained-generation-python

面向本地推理的**受约束生成**服务：在解码推进的每一步就限制可选 token 范围，
而不是生成结束后才校验。任一时刻被选出的候选都仍然可能闭合为符合结构定
义的完整结果——不会走上"中途看似合法、最终无法闭合"的路径。

## 工作原理

```
结构定义 (regex / json_schema)
        │  compile (一次性, 结果缓存)
        ▼
   NFA + 活性剪枝          ← 无法到达接受态的状态被预先剔除
        │  per request
        ▼
  ConstraintState (彼此独立)
        │  每个解码步
        ▼
 allowed_tokens = ⋂ 各约束的合法 token  (∪ EOS, 仅当全部约束可接受)
        │
        ▼
  在 allowed 内取 logits 最大者 → 追加 → 推进各约束状态
```

- 约束统一编译为 NFA（正则子集 → Thompson 构造；JSON Schema 子集 → 先编
  译为等价的"规范化 JSON 文本"正则）。NFA 以状态集合模拟推进，所有无法到
  达接受态的状态在编译期剔除，因此**每一步的可选集合都保证仍可闭合**。
- 编译结果按定义哈希（canonical JSON 的 SHA-256）缓存在有界 LRU 中，**重
  复出现的结构定义不会重复编译**；token 掩码按 NFA 状态集记忆化，候选筛
  选耗时随候选规模与并发数保持有界。
- 每个请求持有独立的 `ConstraintState`，编译产物不可变并跨请求共享；单
  个请求失败（含内部异常）不会影响其他请求。

## API

所有 `/v1/*` 接口需要请求头 `X-API-Key`。每个 API key 映射一个租户；约
束与生成请求严格按租户隔离，**跨租户访问与资源不存在无法区分**（一律
404）。默认开发密钥：`dev-key-a`→`tenant-a`、`dev-key-b`→`tenant-b`，可
用环境变量 `CONSTRAINED_API_KEYS`（JSON 对象 `{"key": "tenant"}`）覆盖。

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/v1/constraints` | 注册约束：`{"kind": "regex"\|"json_schema", "definition": ...}` → 201 `{constraint_id, kind, definition_hash}`；定义非法时 422 |
| GET | `/v1/constraints` | 列出本租户已注册约束 |
| POST | `/v1/generations` | 创建生成：`{"constraint_ids": [...], "prompt": "...", "max_tokens": 256, "timeout_ms": null}` → 202 请求视图 |
| GET | `/v1/generations/{id}` | 查询请求状态与结果 |
| POST | `/v1/generations/{id}/cancel` | 取消；已终结的请求返回 409 |

同一请求可携带多条约束（≤ 8 条），语义为**交集**：候选必须同时满足全部
约束。`max_tokens` ≤ 4096，`timeout_ms` ≤ 120000。

### 错误约定

所有错误（包括参数校验失败）使用统一信封：

```json
{"error": {"code": "...", "message": "...", "reason": "..."}}
```

| HTTP | code | 含义 |
| --- | --- | --- |
| 401 | `UNAUTHORIZED` | 缺少或非法 API key |
| 404 | `NOT_FOUND` | 约束/请求不存在，或属于其他租户 |
| 409 | `CONFLICT` | 对已终结请求执行取消 |
| 422 | `INVALID_REQUEST` | 请求参数非法（含超限） |
| 422 | `INVALID_CONSTRAINT` | 约束定义被拒绝，`reason` 见下表 |
| 500 | `INTERNAL` | 未预期错误 |

### 约束拒绝分类（`error.reason`）

无法满足、彼此矛盾或定义本身非法的结构在**开始生成前**拒绝，绝不静默退
化为不加约束：

| reason | 含义 | 示例 |
| --- | --- | --- |
| `invalid_definition` | 定义无法解析/类型错误 | 正则括号不配对、`{"type": "foo"}` |
| `unsupported` | 使用了不支持的构造 | `oneOf`、`pattern`、开放对象、无 `items` 的数组 |
| `contradictory` | 定义自相矛盾 | `minimum > maximum`、`minLength > maxLength`、空 `enum`、封闭对象里 `required` 未定义的属性 |
| `unsatisfiable` | 良构但不可能匹配任何输出 | schema `false`、编译后无到达接受态的路径 |
| `too_complex` | 超出有界资源限制 | 重复次数 > 512、嵌套 > 32 层、整数界超出 int64 |

### 生成请求终结状态

`completed`（`output_text` 为完整结果）、`failed`（`error.reason` 为
`max_tokens_exceeded` 或 `internal`，不会返回部分结果）、`cancelled`、
`timed_out`。取消与超时在每个解码步检查，取消后尽快释放计算资源。

请求状态仅保存在内存中：**进程重启后未完成的请求不存在，也就永远不会
被当作已完成结果返回**（重启后查询一律 404）。

## 约束语义

### `regex`

支持子集：字面量、`.`、字符类（区间、取反）、转义（`\d \D \w \W \s \S
\n \t \r \f \v \xHH \uHHHH` 及标点转义）、`* + ? {m} {m,} {m,n}`（懒惰
后缀接受但忽略）、分组 `(...)` 与 `(?:...)`、 alternation `|`、锚点
`^ $`（无操作，匹配恒为全串匹配）。

### `json_schema`

输出必须是**规范化 JSON**：无多余空白；对象键顺序为"必填属性（按声明顺
序）在前、出现的可选属性（按声明顺序）在后"；字符串与键使用
`json.dumps(..., ensure_ascii=True)` 形式。

支持：`type`（object/array/string/integer/number/boolean/null）、
`properties`/`required`/`additionalProperties: false`、`items`/
`minItems`/`maxItems`、`minLength`/`maxLength`、`minimum`/`maximum`/
`exclusiveMinimum`/`exclusiveMaximum`（仅 integer，按 int64 区间精确编
译）、`enum`、`const`。

显式不支持（出现即 `unsupported` 拒绝，而非静默忽略）：`oneOf`/`anyOf`/
`allOf`/`not`/`$ref`/`pattern`/`uniqueItems`/`multipleOf`/`contains`/
`patternProperties`/`dependentRequired`/`if`/`then`/`else`/
`minProperties`/`maxProperties`、开放对象（`additionalProperties` 非
`false`）、无 `items` 的数组、`number` 类型的数值界、无 `type` 且无
`enum`/`const` 的任意值模式、布尔形式的 `exclusiveMinimum/Maximum`。

## 运行与测试

```bash
uv sync
uv run uvicorn main:app --reload
uv run pytest
```

模型侧当前接入的是确定性 mock（`constrained.model.MockModel`，接口为
`logits(context, vocab_size)`），替换为真实本地推理模型时无需改动约束
引擎与服务层。
