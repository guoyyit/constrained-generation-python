# 受约束生成（Constrained Generation）

本服务在生成推进过程中逐步限制可选 token 范围，而不是在生成结束后再校验：
任一时刻被选出的候选都保证仍能扩展为符合结构定义的完整输出，不存在
"中途看似合法、最终无法闭合" 的路径。

## 实现原理

1. 每个约束定义被编译为一个**已裁剪（trimmed）的 DFA**：自动机中任意状态
   都保证可达某个接受状态，因此自动机允许的每个 token 都仍然可以闭合出
   合法结果。
2. 同一请求携带多条约束时，对各自 DFA 求**乘积自动机**（交集）并再次裁剪。
   候选集合 = 乘积自动机在当前状态下允许的 token 集合（接受状态时附加
   停止信号）。
3. 生成循环每步将模型 logits 与该候选集合求交后采样，因此输出必然属于
   所有约束语言的交集。
4. 编译结果按定义的规范化形式缓存（LRU），重复出现的结构定义不会重复
   编译；每步的候选筛选是对当前状态的 O(1) 查表，与候选集合规模、并发
   请求数无关。

## 约束定义

请求体 `constraints` 为非空数组（至多 8 条），元素形如
`{"kind": <种类>, "body": <定义体>}`。

### `json_schema`：JSON Schema 子集

支持的类型：`object` / `array` / `string` / `integer` / `number` /
`boolean` / `null`，以及 `enum` / `const`。

| 类型 | 支持的关键字 | 限制 |
| --- | --- | --- |
| object | `properties` `required` `additionalProperties` | 至多 8 个属性；`additionalProperties` 仅允许 `false`；属性按声明顺序生成 |
| array | `items` `minItems` `maxItems` | `items` 必填；上下界不超过 16 |
| string | `minLength` `maxLength` | 上限不超过 64；内容字符为可打印 ASCII（不含 `"` 与 `\`） |
| 通用 | `type` `enum` `const` | `enum`/`const` 只能与 `type` 组合 |

未列出的关键字（`pattern`、`minimum`、`allOf`、布尔 schema 等）属于
**合法但超出子集**，会被拒绝并归类为 `constraint/unsupported`。

生成的 JSON 使用紧凑分隔符（如 `{"a":1}`），均为合法 JSON。

### `regex`：正则方言

对整个输出做全串匹配。支持：字面量、字符类（`[a-z]`、`[^...]`）、
`.`、`\d` `\w` `\s`、分组、`|`、`*` `+` `?` `{m,n}`（上限 64）。
不支持锚点 `^`/`$`（匹配语义本身就是全串）。所有字符必须落在
可打印 ASCII 词表内。

## 失败分类

所有错误返回统一信封：

```json
{"error": {"code": "...", "message": "...", "details": {...}}}
```

### 生成前拒绝（请求不落盘、不消耗生成算力）

| code | HTTP | 含义 |
| --- | --- | --- |
| `constraint/invalid` | 400 | 定义本身非法（缺字段、类型错误、正则语法错误、字符超出词表等） |
| `constraint/unsupported` | 400 | 定义合法但超出支持的子集（如 `pattern`、递归结构、超上限） |
| `constraint/unsatisfiable` | 422 | 定义良构但无任何输出可满足（空 `enum`、`minLength > maxLength` 等） |
| `constraint/contradictory` | 422 | 多条约束彼此矛盾，交集为空（乘积自动机为空语言） |
| `request/invalid` | 400 | 请求参数非法（空约束数组、超界的 `max_tokens`/`timeout_ms`、缺租户头、请求体畸形） |

被拒绝的请求**不会**静默退化为无约束生成。

### 生成期失败（记录在请求状态上）

| code | 状态 | 含义 |
| --- | --- | --- |
| `generation/cancelled` | `cancelled` | 请求被取消；结果为空，工作线程在下一个 token 边界释放 |
| `generation/timeout` | `failed` | 超过 `timeout_ms` 截止时间 |
| `generation/incomplete` | `failed` | `max_tokens` 耗尽时仍未到达可闭合状态 |
| `generation/aborted` | `aborted` | 进程重启时仍处于非终态的请求；**不会**被当作已完成返回 |

### 其他

| code | HTTP | 含义 |
| --- | --- | --- |
| `request/not_found` | 404 | 请求不存在，或不属于当前租户（跨租户访问一律 404，不泄露存在性） |
| `request/conflict` | 409 | 对已终态的请求执行取消 |

## API

所有端点要求 `X-Tenant-Id` 请求头；请求状态仅对创建它的租户可见。

- `POST /v1/generations` — 提交生成请求。
  请求体：`constraints`（必填）、`max_tokens`（1..4096，默认 512）、
  `timeout_ms`（1..60000，默认 10000）、`seed`（默认 0，相同种子与约束
  产生相同输出）、`wait`（默认 true，同步等待终态）。
- `GET /v1/generations/{request_id}` — 查询状态与结果。
- `POST /v1/generations/{request_id}/cancel` — 取消；取消在 token 边界
  生效，计算资源随即释放。

响应：`{"request_id", "status", "result", "error", "created_at",
"updated_at"}`，`status ∈ pending | running | completed | failed |
cancelled | aborted`。

## 隔离与恢复

- 每个请求持有独立的会话、取消事件与约束游标；请求间不共享可变状态。
  单个请求的失败（含约束拒绝、超时、取消）不影响其他请求。
- 编译后的自动机不可变，可安全地被并发请求共享（缓存）。
- 请求状态变更追加写入 JSONL 台账（配置数据目录时）。进程重启后回放
  台账，所有非终态请求被清扫为 `aborted`；已完成请求的结果保留。

## 本地运行

```bash
uv run pytest            # 运行测试
uv run uvicorn main:app  # 启动服务
```
