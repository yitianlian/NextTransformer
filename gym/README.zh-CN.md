# NextTransformer Gym

[English](README.md) · [设计文档](DESIGN.zh-CN.md)

Gym 用于运行模型架构研究实验：宿主机执行训练，Harbor 在独立容器中运行 Claude Code 和提交验证器。
本文示例使用合成数据，不代表生产 ClimbMix 或 CORE-22 成绩。

- [安装](#安装) · [启动研究会话](#启动研究会话) · [运行实验并提交](#运行实验并提交)
- [Gym 接口](#gym-api) · [候选格式](#提交原生候选) · [Worker 隔离](#worker-isolation)
- [错误与恢复](#错误与恢复) · [检查安装](#检查安装)

Agent 的操作要求见[任务指令](tasks/architecture-research/instruction.md)。以下宿主机命令均在仓库根目录执行。

## 安装

环境要求：

- Linux、Docker 和非特权用户命名空间。
- Python 3.12 或 3.13，以及 `uv`。
- bubblewrap，或支持 Landlock 的内核，详见 [Worker 隔离](#worker-isolation)。

```sh
uv sync --project gym --extra worker
docker build -f gym/Dockerfile --target client -t nt-gym-client:local .
docker build -f gym/Dockerfile --target verifier -t nt-gym-verifier:local .
```

`worker` 扩展安装 CPU 运行依赖。GPU 训练需另备 CUDA Python 环境，版本要求见 [pyproject.toml](pyproject.toml)，无需 CUDA 容器镜像。
启动器使用本地已有镜像并记录其不可变 ID；部署时应以审核过的 digest 固定基础镜像。

## 启动研究会话

在控制器环境中设置模型供应商凭据，Harbor 会将其传给 Claude Code。
示例配置指定模型、CLI 版本、时间和轮次上限，以及预算：

| 配置项 | 含义 |
| --- | --- |
| `agent.max_budget_usd` | 可选的 API 费用上限，可用 `--max-budget-usd` 覆盖；显式值必须是大于零的有限数值。省略时不设置上限。 |
| `budget.research_flops_limit` | 会话共享的研究 FLOPs 额度；最终训练使用独立额度。 |

先检查启动配置，再启动会话：

```sh
uv run --project gym python gym/run_trial.py --config gym/examples/claude-code.toml --dry-run
uv run --project gym python gym/run_trial.py --config gym/examples/claude-code.toml
```

使用 DeepSeek Flash 时，选择 `gym/examples/deepseek-v4.1-flash.toml`。
将 `GYM_RESEARCH_MODEL` 设为供应商实际提供的模型 ID。
配置 `ANTHROPIC_BASE_URL` 和 `ANTHROPIC_API_KEY`。接口必须支持 Anthropic Messages。
严格的费用上限应由供应商控制。CLI 美元额度依赖供应商提供的用量和价格信息。

### GPU 配置

先检查 GPU 占用情况，再为已分配的 GPU 创建本地配置：

```sh
mkdir -p .agent-work
cp gym/examples/resources.gpu.json .agent-work/resources.local.json
```

将 `gpu_ids` 设为已分配的 GPU 索引或 UUID。用 `gpu_memory_limit_bytes` 设置每张卡的显存额度。
保持 `max_concurrent_jobs` 为 `1`。然后启动会话：

```sh
uv run --project gym python gym/run_trial.py \
  --config gym/examples/deepseek-v4.1-flash.toml \
  --execution-config .agent-work/resources.local.json \
  --worker-python /path/to/cuda-environment/bin/python
```

CPU 示例可省略 GPU 配置。**同一时间只启动一个研究会话**：每份 Gym 状态只有一个训练执行槽，不同状态目录不共享机器级调度锁。
GPU 忙时作业排队，不会转到 CPU。

<a id="gym-api"></a>

## Gym 接口

Harbor 注册 `gym` MCP 工具；Python `GymClient` 使用同一套 HTTP API。
Agent 容器中的 `/run/gym-auth/credential.json` 提供 `endpoint`、`token` 和 `session_id`。

先调用 `describe_space`，读取操作参数 schema、候选契约和执行配置（profile），再按以下流程操作：

| 阶段 | 常用操作 |
| --- | --- |
| 查看资源与参考资料 | `get_resources`、`list_components`、`read_reference` |
| 查看已有实验 | `list_experiments`、`read_experiment` |
| 提交与检查候选 | `submit_candidate`、`inspect_cost`、`calibrate_candidate`、`validate_candidate` |
| 报价与运行 | `quote_experiment` → `run_experiment` → `get_job` / `get_result` |
| 查看预算与取消作业 | `get_budget`、`cancel_job` |
| 冻结与查询提交 | `finalize`、`get_submission` |

使用时注意：

- `submit_candidate`、`calibrate_candidate`、`validate_candidate`、`debug_candidate`、`run_experiment`、`cancel_job` 和 `finalize` 需要 `idempotency_key`（1–128 个字符）。相同输入重试时复用原键，输入改变时使用新键。
- 报价不预留 FLOPs；`run_experiment` 才预留报价额度。
- `target_update` 是绝对目标步数，默认 1，不能超过 profile 上限；续训时必须大于检查点步数。
- 资源 ID 从 `get_resources` 获取；省略资源选择参数时，Gym 自动选择已分配的空闲 GPU。
- 实验查询默认读取公开实验；查询本会话需设置 `scope="session"`。

完整操作、HTTP 调用、分页和诊断参数见[接口参考](README.md#gym-api)。

## 提交原生候选

候选包含原生插件源码，以及部分或完整的原生 `ModelConfig` 字段。
ZIP 结构如下：

```text
proposal.json
architecture.json
plugin/<method>/method.json
plugin/<method>/__init__.py
plugin/<method>/<optional local Python modules>
```

ZIP 文件和解压后的总内容分别不能超过 1 MiB。最多包含 32 个 UTF-8 文本文件，不要加入目录条目。
不要加入权重、安装程序或训练脚本。上传时，Gym 保存源码，但不导入源码。
打包示例见 [make_candidate.py](examples/make_candidate.py)。

候选包的关键约束：

| 文件或字段 | 要求 |
| --- | --- |
| `proposal.json` | 必填 `schema_version="gym.candidate.v1"`、`sdk_version="reference_plugin_v1"` 和 `method`。 |
| 方法名称 | 与插件目录及 `method.json.name` 一致，不得覆盖已注册方法。 |
| `architecture.json` | 使用原生 `ModelConfig` 字段和 `method.json.options` 声明的选项。 |
| profile 固定字段 | `width`、`heads`、`sequence_len`、`vocab_size`、`scale` 可省略；显式值必须与 profile 一致。 |
| `routing_seed` | 省略或设为 1729。 |
| 原生 `model` hook | 必须继承 `ReferenceLM`；接口见 `plugin-contract` 和 `reference-model` 参考资料。 |

模型配置不接受训练设置、路径、命令或环境变量。整数字段不接受字符串或布尔值，数值必须有限，不要用 `null` 代替省略字段。
字段默认值、选项类型及合并顺序见[候选契约](README.md#submit-a-native-candidate)；最终仍须通过原生检查。

只改配置时，用原源码和新的 `architecture.json` 再次上传。
也可以向 `calibrate_candidate.grid` 传入 1–32 个完整的 architecture 对象。
每个对象会完整替换父候选的 architecture。
对返回的作业 ID 调用 `get_result`，读取 `derived[].candidate` 和 `derived[].cost`。
校准不会训练模型或测量吞吐。源码或配置改变后会生成新候选，新候选不能沿用父候选的检查点。

## 运行实验并提交

在 Agent 容器中使用下面的示例。先创建候选包：

```sh
python /opt/gym/make_candidate.py /workspace/candidate.zip
```

下面的 profile 名称对应合成数据示例。

```python
import json
from pathlib import Path
from nexttransformer_gym import GymClient

binding = json.loads(Path("/run/gym-auth/credential.json").read_text())
gym = GymClient(binding["endpoint"], binding["token"])
candidate = gym.submit_candidate(Path("/workspace/candidate.zip").read_bytes(), "idea-1")
quote = gym.quote_experiment(
    candidate_id=candidate["id"], profile_id="research-fixture", target_update=2
)
job = gym.run_experiment(quote_id=quote["id"], idempotency_key="run-1")
result = gym.wait_result(job["id"])
checks = gym.validate_candidate(
    candidate_id=candidate["id"], profile_id="final-fixture", idempotency_key="checks-1"
)
assert gym.wait_result(checks["id"])["admitted"]
submission = gym.finalize(
    candidate_id=candidate["id"], profile_id="final-fixture", idempotency_key="final-1"
)
Path("/workspace/submission.json").write_text(json.dumps({
    "session_id": binding["session_id"],
    "submission_id": submission["id"],
    "candidate_hash": candidate["hash"],
}))
```

SDK 的 `submit_candidate` 接收 ZIP 字节；HTTP 调用需将其编码为 base64，传入 `package_base64`。
返回的 `architecture` 是上传的部分配置，Gym 会按所选 profile 补齐默认值。

读取结果前先检查作业状态。结果尚未生成时，`get_result` 返回 `not_found`。
执行成功的检查作业仍可能包含未通过的检查。提交前读取 `checks` 和 `admitted`。
调用 `finalize` 前，先结算或取消所有作业。账户不能有未结算的预留额度。
Gym 保留源码。验证器只检查提交回执，不启动最终训练。
资源选择示例见 [research_with_placement.py](examples/research_with_placement.py)。

<a id="worker-isolation"></a>

## Worker 隔离

候选代码在宿主机的 worker 进程中运行。在执行配置中用 `isolation` 选择隔离后端：

| 取值 | 隔离方式 | 适用场景 |
| --- | --- | --- |
| `bwrap`（默认） | 新的 bubblewrap 用户、挂载、PID 和网络命名空间 | 允许非特权挂载命名空间的宿主机 |
| `landlock` | Landlock 路径规则，以及用户、网络、IPC 和 UTS 命名空间。CPU worker 另有 PID 命名空间和禁止新建 Unix socket 的 seccomp 过滤器 | 禁止挂载的宿主机，例如非特权 Kubernetes Pod |
| `none` | 不隔离 | 仅限可信候选。Gym 启动时打印警告 |

`landlock` 要求控制器以 root 运行，并需要 util-linux `unshare` 以及支持 Landlock 的 x86_64 或 aarch64 内核。
CPU 会话可以只配置这一个字段：

```json
{"schema_version": "gym.execution.v1", "isolation": "landlock"}
```

`landlock` 的隔离弱于 `bwrap`：

- worker 能判断路径是否存在，并能列出平台根目录和 Gym 源码根目录下的所有目录，但不能读取允许范围以外的文件。
- Landlock 不限制 chmod、utime 等元数据修改。
- CUDA 需要宿主 PID 命名空间和 Unix socket，因此 CUDA worker 保留两者，其脱离的进程可能在任务结束后继续运行。
  以 `nobody` 运行时，它们只能连接所有用户都可写的 Unix socket。
- CUDA worker 能打开所有 NVIDIA 设备节点。`CUDA_VISIBLE_DEVICES` 选择分配的 GPU，但不把 worker 限制在这张卡上。

控制器以 root 运行时，所有后端的 worker 都以 `nobody` 运行。该用户必须能读取平台源码和 worker Python。
`get_resources` 在 `execution_config` 中报告当前后端。

## 错误与恢复

HTTP 的 `error` 对象包含 `code`、`hint`、`phase`、`retryable` 和 `details`。
SDK 抛出 `GymError`。MCP 保留原错误，并设置 `isError=true`。

| 错误码 | 含义 |
| --- | --- |
| `invalid_request` | 操作参数不符合 API schema 或请求规则。 |
| `invalid_artifact` | 包格式、模型字段名称或字段类型不正确。 |
| `incompatible_contract` | proposal 方法或路由随机种子不符合约定。 |
| `conflict` | 相同幂等键对应了不同输入，或发生其他状态冲突。 |
| `incomplete_coverage` | 静态审计失败。该路径中的原生配置或 profile 检查失败也使用此错误码。 |

保留会话的宿主状态、源码版本、profiles 和 worker Python 环境。
运行控制器命令时，使用冻结的 `<trial>.host/execution.json`。该文件已将 GPU 索引转换为 UUID。
保留原 server ID。运行 `serve` 时，还需通过 `--host` 指定原绑定地址。监听配置只保存端口。

崩溃后，先确认 worker 已退出，再运行受信任的控制器命令。将占位符替换为实际路径和 ID：

```sh
gym/.venv/bin/python -m nexttransformer_gym.cli \
  --state <retained-state> --profiles <profiles.json> --platform . \
  --execution-config <trial>.host/execution.json --worker-python <worker-python> \
  reconcile --session-id <owner-id> --job-id <job-id> --worker-confirmed-stopped
```

若其他调度器仍持有执行槽，控制器会拒绝结算。结算会扣除剩余的保守预留额度。
重复执行不会重复扣费。确认参数只声明 worker 已退出，不会终止进程。
通过 `gym/.venv/bin/python -m nexttransformer_gym.cli --help` 查看控制器命令。
命令包括 `prepare-final`、`launch-final`、`work-once`、`get-final`、`snapshot` 和 `serve`。
只有受信任的控制器命令才能分配独立额度并启动最终训练。

## 检查安装

运行测试、变更检查和本地集成检查：

```sh
PYTHONPATH=gym/src:. NANOCHAT_DTYPE=float32 TORCHDYNAMO_DISABLE=1 \
  uv run --project gym --extra worker python -m pytest -q gym/tests
PYTEST_XDIST_AUTO_NUM_WORKERS=2 uv run --extra cpu python -m pytest -q -n auto
gym/.venv/bin/python -m nexttransformer.checks.changes --base origin/main
uvx --from ruff==0.14.10 ruff check --select F gym
uvx --from ruff==0.14.10 ruff format --check gym
uv run --project gym python gym/smoke.py
```

Smoke 使用原生 Claude Code 和本地模型桩，不调用外部模型。
如果客户端镜像没有固定版本的 CLI，Harbor 会安装它。
安装站点不可达时，通过 `--client-image <image>` 指定已包含该版本 CLI 和 Gym 客户端文件的镜像。

进行短 GPU 检查时，选择已分配的显卡：

```sh
PYTHONPATH=gym/src:. gym/.venv/bin/python gym/gpu_smoke.py \
  --state-dir .agent-work/gpu-smoke \
  --gpu <allocated-index-or-uuid> --worker-python <cuda-python>
```

这些检查验证集成行为。生产数据和最终基准成绩需要另行验收，详见[设计限制](DESIGN.zh-CN.md)。
