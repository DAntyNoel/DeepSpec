# DeepSpec 中文说明

本仓库用于复现、训练和评估 DSpark / DFlash / Eagle3 draft model。

- `origin`：个人 fork，用于保存本地适配和实验脚本。
- `upstream`：原作者仓库，用于持续接收官方更新。

## 当前 fork 差异

本 fork 相对 upstream 主要包含以下几类差异：

- 本地数据和输出忽略规则：`local_data`、`local_outputs/` 不进入 git。
- `eval.py` 支持 `--dataset-root` 和 `--tasks`，便于在本地只跑小样本任务。
- 数据下载脚本支持 `--skip-invalid`，可跳过格式异常的 conversation。
- MTS 本地 DSpark Qwen3 配置和脚本：`config/dspark/dspark_qwen3base_4b_local.py`、`scripts/*_local.*`。
- 不依赖真实模型和数据的 smoke 脚本：`scripts/smoke_dspark_minimal.py`、`scripts/smoke_dspark_decode.py`。

这些差异主要服务于 MTS / pjlab 环境下的复现和验证。带有本地路径、Slurm 分区、模型目录假设的脚本适合保留在个人 fork，不适合直接提交给 upstream。


## 常用命令

运行无需真实模型权重的 smoke：

```bash
PYTHONPATH=. ../../.venv/bin/python scripts/smoke_dspark_minimal.py
PYTHONPATH=. ../../.venv/bin/python scripts/smoke_dspark_decode.py
```

下载本地 PerfectBlend 数据：

```bash
bash scripts/data/download_perfectblend_local.sh
```

本地小样本解码评估：

```bash
bash scripts/eval/eval_dspark_local.sh
```

统一复现入口：

```bash
bash scripts/reproduce_dspark_local.sh smoke
bash scripts/reproduce_dspark_local.sh decode-smoke
bash scripts/reproduce_dspark_local.sh train
bash scripts/reproduce_dspark_local.sh eval
```

## 提交建议

推荐把 fork 内部提交拆成清晰的逻辑：

1. upstream 同步提交：只包含上游 fast-forward 或合并结果。
2. 通用功能补丁：如 `--dataset-root`、`--tasks`、`--skip-invalid`、smoke 脚本。
3. MTS 本地适配：本地 config、Slurm 脚本、复现实验入口。

提交前至少执行：

```bash
git diff --check
git diff --cached --check
../../.venv/bin/python -m compileall <changed-python-files>
PYTHONPATH=. ../../.venv/bin/python scripts/smoke_dspark_minimal.py
PYTHONPATH=. ../../.venv/bin/python scripts/smoke_dspark_decode.py
```

如果只改文档，可跳过 smoke，但仍应检查 `git diff --check`。

不直接 PR upstream。
