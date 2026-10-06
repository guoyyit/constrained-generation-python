# constrained-generation-python

面向本地推理的受约束生成服务：在生成推进过程中按输出结构定义限制可选
token 范围，保证任一中间前缀都能闭合为合法结果。

约束语义、失败分类与 API 说明见
[docs/constrained-generation.md](docs/constrained-generation.md)。

```bash
uv run pytest            # 测试
uv run uvicorn main:app  # 服务
```
