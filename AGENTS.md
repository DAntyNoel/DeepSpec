# 重要提示

本文件约束在 `3rd/DeepSpec` 内工作的自动化代理和协作者。若与 MTS 主仓库的全局规则冲突，以 MTS 主仓库规则为准；若只涉及 DeepSpec fork 内部维护，以本文件为补充。

## 仓库定位

- 本目录是独立 git 仓库，也是 MTS 主仓库计划跟踪的第三方 submodule。
- `origin` 应指向个人 fork：`git@github.com:DAntyNoel/DeepSpec.git`。
- `upstream` 应指向原作者仓库：`git@github.com:deepseek-ai/DeepSpec.git`。
- 不要把 DeepSpec 文件复制到 MTS 主仓库中；MTS 只跟踪 submodule commit 指针。

## 分支和同步

- 默认在 DeepSpec 的 `main` 分支维护 fork 差异。
- 需保持接收原作者更新。
- 如果 upstream 更新与 fork 修改冲突，先列出冲突文件和上下游意图，再解决；不要用重置或覆盖方式丢弃本地适配。
- 同步 upstream 后，应重新运行与改动相关的 smoke 或编译检查。

## 文件管理

允许提交：

- fork 必需的轻量配置文件。
- MTS 本地复现、下载、评估脚本。
- 不依赖真实模型和数据的 smoke 脚本。
- 对 upstream 默认行为兼容的通用小补丁。
- 文档和仓库管理说明。

禁止提交：

- `local_data`。
- `local_outputs/`。
- checkpoint、TensorBoard event、Slurm 输出、模型权重、数据集 cache。
- 私有 token、账号、机器密钥。
- 与当前任务无关的大规模格式化或重排。

带有 MTS 或 pjlab 路径的脚本必须提供环境变量覆盖方式，不能只写死绝对路径。

## 本地环境

- 默认使用 MTS 根目录的 Python：`../../.venv/bin/python`。
- 运行 DeepSpec 脚本时通常需要 `PYTHONPATH=.`。
- 下载 Hugging Face 数据或模型时，优先使用镜像站和本地 cache；不要把 cache 写入仓库跟踪目录。
- 本地输出默认放在 `local_outputs/`，数据/cache 默认放在 `local_data/`。

## 修改原则

- 优先保持 upstream 兼容：新增参数应有默认值，并保持原始默认行为。
- MTS 专用逻辑应放在 `*_local.*` 脚本或本地 config 中，不要侵入核心训练/评估代码。
- 若必须修改核心代码，应说明原因、影响范围和验证结果。
- 对第三方 upstream 代码不要做无关重构，避免后续同步冲突扩大。

## 验证

提交前根据改动范围选择验证。

文档或脚本说明改动：

```bash
git diff --check
git diff --cached --check
```

Python 代码改动：

```bash
../../.venv/bin/python -m compileall <changed-python-files>
```

DSpark 训练/解码路径改动：

```bash
PYTHONPATH=. ../../.venv/bin/python scripts/smoke_dspark_minimal.py
PYTHONPATH=. ../../.venv/bin/python scripts/smoke_dspark_decode.py
```

真实模型、数据下载或 Slurm 任务可能耗时较长，提交前不要默认启动；除非用户明确要求，先做轻量 smoke。

## 提交和推送

- commit 应先在 DeepSpec 子仓库内完成，再回到 MTS 主仓库更新 submodule 指针。
- DeepSpec fork 的 commit 信息应清楚区分：
  - upstream sync；
  - 通用功能补丁；
  - MTS 本地适配；
  - 文档更新。
- 推送目标是个人 fork 的 `origin`，不要直接向原作者 `upstream` 推送。
- MTS 主仓库提交 submodule 指针前，应确认 DeepSpec 内部没有未处理冲突或未提交的必要代码。

