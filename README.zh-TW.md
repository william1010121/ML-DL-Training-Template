# ML/DL Training Template

**一套重視證據與可重現性的 PyTorch 訓練模板：設計目標涵蓋 native、Docker 與 Apptainer HPC，同時不讓訓練程式綁定平台或 tracker。**

[English](README.md)

很多訓練專案很容易開始，六週後卻很難說清楚結果從哪裡來。這份模板把 experiment config 當成「意圖」、run manifest 當成「證據」、research note 當成「決策紀錄」。資料、權重、logs 與 checkpoints 留在 Git 外；重現結果所需的小型事實則保留成可 review 的文件。

模板包含真的 MNIST 參考專案、嚴格 Pydantic config、local-first metrics、固定的 `mltrain` CLI、只封裝環境的 container，以及用來安全維護研究流程的 repo-local Codex skills。

## 為什麼值得用

- **一個實驗，一份完整 config。** 沒有隱藏繼承，也沒有某台機器專用的絕對路徑。
- **沒有證據就不宣稱結果。** Provenance 與 validation 未通過前不能 promote。
- **乾淨是預設行為。** Dataset、checkpoint、完整 run、SIF 與 secret 都不進 Git。
- **只維護一份環境。** Docker 是 source of truth，Apptainer 使用同一個 immutable OCI image。
- **Tracker 是 optional。** 本機 JSONL 與 manifest 永遠存在，外部 tracker 只是 adapter。
- **適合 AI agent 協作。** Repo-local skills 維護實驗編號、研究紀錄與依賴邊界。

## 五分鐘 CPU 開始

需要 Git、Python 3.11–3.13 與 [uv](https://docs.astral.sh/uv/)。MNIST 必須明確下載；訓練不會暗中連網，也不會偷偷換成假資料。

<!-- sync:start quickstart -->
```bash
uv sync --extra cpu
uv run python scripts/download_mnist.py --root datasets
uv run mltrain train --config configs/mnist-baseline/exp-001.yml
```
<!-- sync:end quickstart -->

結果會寫到 `runs/mnist-baseline/exp-001/<run-id>/`。先看 `result.json`，再明確驗證：

<!-- sync:start validate-command -->
```bash
uv run mltrain validate --run runs/mnist-baseline/exp-001/<run-id>
```
<!-- sync:end validate-command -->

Validation 只會把 run 判定為 `exploratory` 或 `completed`，不會暗中修改研究紀錄。有明確決策後再入帳：

<!-- sync:start record-command -->
```bash
uv run mltrain record-result \
  --run runs/mnist-baseline/exp-001/<run-id> \
  --decision "Keep as the CPU baseline"
```
<!-- sync:end record-command -->

如果 strict、clean、completed run 應成為 tracked evidence，就改用：

<!-- sync:start promote-command -->
```bash
uv run mltrain promote \
  --run runs/mnist-baseline/exp-001/<run-id> \
  --decision "Promote as the CPU baseline"
```
<!-- sync:end promote-command -->

Promotion 只會把可 review 的小型 metadata 寫進 `artifacts/`，不會提交 dataset 或 checkpoint。

## 目錄地圖

```text
.
├── .agents/skills/               # Repo-local 專案治理流程
├── src/
│   ├── mltrain/                   # 固定 CLI、contract、provenance gate
│   └── ml_training_template/      # 可替換的 project adapter
│       ├── data/
│       ├── model/
│       ├── training/
│       ├── validate/
│       └── tracking/
├── configs/
│   ├── research.md                # 專案目標、研究線索引、全域決策
│   ├── registry.yml               # 實驗生命週期與證據指標
│   └── mnist-baseline/
│       ├── research.md            # 這條研究線的目標與精簡結果
│       └── exp-001.yml … exp-003.yml
├── datasets/                      # 本機資料；內容忽略
├── checkpoints/                   # 本機／預訓練權重；內容忽略
├── runs/                          # 完整本機 run；忽略
├── artifacts/                     # Promoted 小型證據；追蹤
├── containers/                    # CPU/CUDA 環境定義
├── scripts/                       # 下載與 runtime helpers
└── tests/                          # Contracts、smoke tests、repo hygiene
```

這個切法是刻意的：`src/.../data` 與 `src/.../model` 放程式，`datasets/` 與 `checkpoints/` 放本機檔案。`configs/research.md` 是全專案研究地圖，每個 `configs/<research-line>/research.md` 才負責該研究線的結果。

Research-line slug 是 folder、每份 config 的 `experiment.research_line`、`configs/registry.yml` 與 top-level research index 之間的 join key。Registry entry 保存 machine-readable 的 `config`、`status`、鎖定 config hash、completed run 與 promoted artifact pointers；該研究線的 `research.md` 則是人類閱讀的結果帳本。請保持兩邊一致，不要另外複製一套 metadata。

## 實驗生命週期

```text
planned config
    │
    ├─ train ───────> ignored run + manifest + metrics
    │                        │
    │                     validate
    │                        │
    └──────────────> exploratory or completed
                             │
                    record-result or promote
                             │
                 registry + line research + evidence
```

實驗一旦入帳，config 就不可再改。Seed、dataset、model 或訓練策略有變化，請建立下一個 `exp-###.yml`，不要改寫歷史。失敗或中斷的 run 可以保留作為本機診斷，但不會成為研究結論。

每次 run 都會記錄 resolved config、source revision 與 dirty 狀態、command、環境 identity、seed、determinism flags、資料／模型 identity、canonical metrics、result 與 validation 狀態。Secret value 永遠不會寫進紀錄。

## 建立自己的專案

先用 GitHub Template 建立 repository 並完成一次 commit。初始化會刪除與改寫 template files，所以只能在 clean worktree 執行；`git status --short` 必須沒有輸出。接著預覽一次性操作：

<!-- sync:start initialize-command -->
```bash
git status --short
python .agents/skills/training-manager/scripts/initialize_project.py \
  --project-name "My Project" \
  --package-name my_project

python .agents/skills/training-manager/scripts/initialize_project.py \
  --project-name "My Project" \
  --package-name my_project \
  --apply
```
<!-- sync:end initialize-command -->

第一個命令只做 dry-run；看過 diff 後才執行帶 `--apply` 的版本。Initializer 會更名 project package、移除 MNIST implementation 與 configs、清除 template adapter 設定，並重設 research registry。它刻意留下尚不可執行的 task shell，不會猜測你的 domain interface。

初始化後按這個順序完成專案：

1. 實作 project config、data、model、training、validation 與 `ProjectAdapter`，並在 `pyproject.toml` 恢復 `[tool.mltrain].adapter`。
2. 用下方 skill scripts 建立有意義的 research line 與完整第一份 experiment config。
3. 依新的 project definition 重新產生並安裝環境。
4. 把 README 中 MNIST-oriented quickstart 與 support claims 換成新任務的真實命令與 evidence；初始化不會替 domain 撰寫文件。

<!-- sync:start post-initialize-command -->
```bash
uv lock
uv sync --extra cpu
uv run python scripts/validate_repo.py
```
<!-- sync:end post-initialize-command -->

做完這些步驟，新的 project 才應被期待可以執行。Template initializer 是安全地移除範例耦合，不是 application generator。

建立研究線與後續實驗也走 deterministic workflow：

<!-- sync:start scaffold-commands -->
```bash
python .agents/skills/training-manager/scripts/new_research_line.py baseline \
  --goal "Establish a reproducible baseline" --apply

python .agents/skills/training-manager/scripts/new_experiment.py baseline \
  --config path/to/full.yml --apply

python .agents/skills/training-manager/scripts/new_experiment.py baseline \
  --from configs/baseline/exp-001.yml --apply
```
<!-- sync:end scaffold-commands -->

這些命令永不覆寫檔案。Codex 會從 `.agents/skills/` 發現 skills；`AGENTS.md` 則規定什麼情況必須使用它們。

## Native、Docker 與 Apptainer

### Docker

Image 只有 lock 過的環境，沒有 application source。Helper 把 repository read-only bind 到 `/workspace`，再把 `datasets/`、`checkpoints/`、`runs/`、`artifacts/` 各自掛成可寫；修改 Python 不需要 rebuild image。

<!-- sync:start docker-commands -->
```bash
docker build -f containers/Dockerfile --target cpu -t ml-training-template:cpu .
./scripts/docker-run cpu ml-training-template:cpu \
  train --config configs/mnist-baseline/exp-001.yml

docker build -f containers/Dockerfile --target cuda -t ml-training-template:cuda .
./scripts/docker-run cuda ml-training-template:cuda \
  train --config configs/mnist-baseline/exp-002.yml
```
<!-- sync:end docker-commands -->

CUDA helper 會加上 `--gpus all`。CUDA 採 fail-closed：指定 CUDA 卻沒有相容、可見的 GPU 就直接失敗，不會偷偷退回 CPU。

啟動前，Docker helper 會把 local tag 解析成 Docker 的 content-addressed image/config ID（`sha256:...`），以該 ID 執行，並把它記進 manifest。這個 local image ID 不是 registry OCI manifest digest；發布到 registry 的 release 另有自己的 immutable OCI digest，Apptainer pull 使用的是後者。

### Apptainer / SingularityCE

先發布 immutable OCI image，再把 `OCI_IMAGE` 設成完整的 `docker://...@sha256:...` URI。不要使用 `latest` 或其他 mutable tag。

<!-- sync:start apptainer-commands -->
```bash
./scripts/apptainer-run pull "$OCI_IMAGE" training.sif

./scripts/apptainer-run exec cuda training.sif \
  train --config configs/mnist-baseline/exp-002.yml
```
<!-- sync:end apptainer-commands -->

Pull helper 會同時記錄來源 OCI digest（`.oci-digest`）與產生的 SIF checksum（`.sha256`），執行前會驗證兩份 sidecars。專案刻意不維護第二份 `Singularity.def`：Docker/OCI 是唯一環境定義。Host 必須提供相容的 NVIDIA driver，並以 HPC site policy 為準。

單機雙卡時，兩個 helpers 都可以啟動 `torchrun`。每次 launch 都要使用唯一、filesystem-safe 的 run ID；helper 會以 `MLTRAIN_RUN_ID` 傳給所有 ranks、建立 single-node rendezvous，並拒絕重用已有 run directory。以下 Docker 與 Apptainer 是 supported contract，但目前仍為 manual/unverified。

<!-- sync:start ddp-command -->
```bash
DOCKER_RUN_ID="exp-003-docker-$(uuidgen | tr '[:upper:]' '[:lower:]')"
./scripts/docker-run cuda ml-training-template:cuda \
  --launcher torchrun --nproc-per-node 2 --run-id "$DOCKER_RUN_ID" -- \
  train --config configs/mnist-baseline/exp-003.yml

APPTAINER_RUN_ID="exp-003-apptainer-$(uuidgen | tr '[:upper:]' '[:lower:]')"
./scripts/apptainer-run exec cuda training.sif \
  --launcher torchrun --nproc-per-node 2 --run-id "$APPTAINER_RUN_ID" -- \
  train --config configs/mnist-baseline/exp-003.yml
```
<!-- sync:end ddp-command -->

Repository 會檢查 `WORLD_SIZE=2` 與 config 是否一致，並拒絕缺少 shared run identity 的 DDP。這些 helpers 只支援 single-node CUDA DDP；multi-node Slurm orchestration 不在 v1 contract 內。如果沒有 `uuidgen`，請自行提供其他唯一、符合 `[A-Za-z0-9_.-]+` 且最長 80 字元的值。

## 支援狀態

「Configured」代表 contract 與 automation 已存在；「Verified」只留給 repository 裡真的有 evidence 的執行。

<!-- sync:start support-matrix -->
| Runtime | Architecture | Status | Evidence |
| --- | --- | --- | --- |
| Native CPU / MNIST | macOS arm64, Linux amd64 | Pending local verification | No promoted run yet |
| Native MPS | macOS arm64 | Manual / unverified | No promoted run yet |
| CPU Docker | Linux amd64 | CI-configured / locally unverified | Docker daemon unavailable during bootstrap |
| Single NVIDIA GPU | Linux amd64, CUDA 12.6 | Supported contract / manual-unverified | No GPU run in this repository |
| Single-node DDP | Linux amd64, 2× NVIDIA GPU | Supported contract / manual-unverified | No DDP run in this repository |
| Apptainer / SIF | Linux amd64 | Supported contract / manual-unverified | Apptainer unavailable during bootstrap |
| Multi-node Slurm | — | Out of scope for v1 | No launcher contract |
<!-- sync:end support-matrix -->

這張表只能跟著新 evidence 更新。初始 CPU MNIST 必須從 clean commit 執行、validate 並 promote 後，才會被寫成 verified。

## 可重現性契約

- Config 是完整 YAML，並用 strict Pydantic model 驗證（`extra="forbid"`）。
- `strict` mode 要求 deterministic 行為，不支援時直接失敗；`performance` mode 明確屬於 exploratory，不能 promote。
- Device request 採 fail-closed；runner 不會自行把 CUDA 或 MPS 改成 CPU。
- `train` 產生本機證據，`validate` 判斷證據，只有明確決策才會修改 tracked research record。
- Result 裡的 source commit 是產生 run 的 commit，不是後來提交文件的 commit。
- `runs/` 保存完整本機紀錄；`artifacts/` 只保存 promoted summary、manifest、checksum 與小型圖表。
- Data、weights、SIF、secrets、mutable image tags 與 host-specific absolute path 都不進 Git。
- 外部 tracker 可以 warning 後降級；canonical local logging 仍是權威紀錄。

## 專案邊界

`mltrain` 是固定、project-neutral 的核心。可替換 package 只實作一個由 `pyproject.toml` 中 `[tool.mltrain]` 選定的 `ProjectAdapter`。Project training 可以依賴自己的 data、model 與 tracking；project validation 不得依賴 training orchestration。換研究任務時不需要 fork 治理核心。

模板刻意採 PyTorch-first，也刻意不預裝 Hydra、Lightning、W&B、MLflow、雲端廠商 API、DVC 或 Git LFS。專案真的需要時再加；`add-experiment-logging` skill 定義了 local-first adapter contract。

## 貢獻

變更應保持小而且有證據。提出修改前先跑 repository checks；不要原地修改已入帳的 experiment，也不要在沒有對應 validated artifact 時升級 support claim。人類與 coding agents 都遵循 `AGENTS.md`。

## 授權

[MIT](LICENSE) © 2026 ML/DL Training Template contributors.
