# STRUCTURE.md

本檔是唯一的架構契約。程式碼與本檔衝突以本檔為準；改架構先改本檔，並在文末「變更記錄」加一則。

這個 repo 只回答三個問題。任何新增的東西都要能說出「拿掉它，哪一題答不出來」，說不出來就不加。
1. 怎麼跑一個實驗：改 config → 執行 → 得到指標。
2. 任一結果由哪份 config、哪個 commit、哪些程式碼、哪顆 seed 產生。
3. 算過的不重算、下載過的不重下載；資料與權重不進 git。

## 架構

```
pipeline/core.py       # stage 快取、依名字載入、遠端同步。無任務邏輯
pipeline/protocols.py  # 下表五個介面的 typing.Protocol
pipeline/{source,preprocess,model,algorithm,evaluate}/   # 五個「軸」，一檔一變體，檔名即名字
configs/research.md    # line 索引：`- <line> — open|concluded — 目標 — 結論`
configs/<line>/goal.md # 四段：問題 / 固定軸與變動軸 / 判勝指標 / 結論
configs/<line>/base.yml, exp-NNN.yml
cache/                 # gitignored，所有 stage 輸出
runs/<line>/exp-NNN/   # resolved.yml + stages.json，只是指向 cache 的指標
run.py  summarize.py  tests/
```

| 軸 | 必要函式 |
|---|---|
| source | `run(out, params)` → 寫資料 + `manifest.json` |
| preprocess | `run(out, params, source, refs)` → 寫資料 + `manifest.json` |
| model | `build(params, manifest) -> nn.Module` |
| algorithm | `train(model, data, out, params, seed, smoke, refs)`：每 epoch append `metrics.jsonl`；寫 `ckpt_last.pt`，有就 resume；結束寫 `ckpt.pt`；`smoke` 時 1 epoch 少量樣本 |
| evaluate | `score(model, data, params, out) -> dict[str, float]`；model 由 core 依 train 的 `meta.json` 重建並載入 `ckpt.pt`；可把預測等附屬檔寫入 `out`，指標一律以回傳值為準 |

- 變體只能 import 標準庫、pyproject 已列套件、同軸 `_common.py`、`protocols`。禁止跨軸、禁止 import core。
- 邊界有兩種：source、preprocess、train、evaluate 四個 stage 之間只透過磁碟目錄格式溝通；train stage 內的 model→algorithm、evaluate stage 內的 model→score，以及 core 依 ref 重建後傳入的模型，是唯一允許的 Python 物件邊界。物件一律是由 core 建立並傳入的通用 `nn.Module`；preprocess、algorithm 與 evaluate 不得 import 具體 model 變體，也不得依 model 名稱寫分支。
- 若 algorithm 或 evaluate 需要模型的特殊能力（例如與結構綁定的 loss、自訂 predict），須在 `protocols.py` 定義 `@runtime_checkable` 的小 Protocol（如 `HasLoss.loss(batch) -> Tensor`），並由變體在模組頂層宣告 `REQUIRES_MODEL = HasLoss`；core 以 `isinstance` 在 stage 執行前檢查。無法以通用 Protocol 表達，就代表它不是可自由組合的獨立變體。
- 跨實驗引用：exp 的 `train`／`preprocess` 可宣告 `refs: {<名>: <line>/exp-NNN}`。core 展開被引用 exp、定位其 train 產物，依其 `meta.json` 重建模型、載入 `ckpt.pt`、`eval()` 並凍結，以 `refs`（dict，名→`nn.Module`）傳入變體。需要 ref 的變體在模組頂層宣告 `REQUIRES_REFS = ("teacher",)`，core 執行前比對 config，不合即失敗；未宣告者收到空 dict。ref 模型同樣受 `REQUIRES_MODEL` 能力檢查。
- `train.init_from: <line>/exp-NNN` 由 core 處理：build 之後、呼叫 train 之前，把該 exp 的 `ckpt.pt` 以 strict 模式載入新 model，不合即失敗；algorithm 不知情。
- `manifest.json` 至少含 `format` 與 `splits`。變體可宣告 `ACCEPTS: dict`，core 執行前比對，不合即失敗。model 只能從 manifest 得知資料形狀。
- 新行為 = 新檔案。修 bug = 改檔，然後重跑所有 stale 的 run。

## Stage 與快取

四個 stage：source → preprocess → train(model+algorithm+seed) → evaluate。
- 目錄 `cache/<stage>/<name>-<key>`，`key = sha256(canonical_json({stage, src, params, up, cache_version}))[:12]`；src = 變體檔全文 + `_common.py` 全文（train 為 model+algorithm 兩者串接）；up = 直接上游目錄名 + 各 ref／`init_from` 解析出的目錄名（排序）；train 的 name 為 `<model>+<algorithm>`。
- ref 解析：展開被引用 exp 的 config → 依本節規則算出其 train 目錄名 → 本機有 `DONE` 即用；否則遠端有即拉取；否則遞迴執行其所需 stages（命中即跳過）。引用構成 DAG，發現循環即失敗。
- 協定：`DONE` 存在→跳過；否則遠端有 `DONE`→拉取；否則在 `<dir>.tmp` 執行（train 可 resume，其他清空重做）→ core 寫 `meta.json`（train）→ 寫 `DONE` → rename → 推遠端，`DONE` 最後推。失敗保留 `.tmp`。
- train 的 `meta.json` 至少記錄 model name、完整 model params、preprocess manifest 與 checkpoint 檔名，供 evaluate 重建同一模型並載入權重。
- evaluate 的回傳值由 core 驗證為有限浮點數，並以 canonical JSON 寫入 evaluate stage 目錄的 `metrics.json`。
- 遠端只靠環境變數 `MLCACHE_REMOTE`（rclone 路徑）與 `rclone lsf|copy|copyto|cat`；未設即純本機。
- `CACHE_VERSION` 在 core.py 頂端；core 改動可能影響輸出時遞增。

## Config 與執行

- exp 是完整 YAML：五軸各 `{name, params}` + `seed`；可 `extends: base.yml`（同目錄、一層、深合併）。`preprocess` 節點可加 `refs`；頂層可加 `train:` 區塊，含 `init_from` 與／或 `refs`（train stage 是 model+algorithm 的組合，引用屬於 stage 而非單一軸）。引用值一律是 `<line>/exp-NNN`（exp config 引用，不是 cache 目錄名）。Pydantic `extra="forbid"`，name 與引用必須對應存在的檔案。
- `python run.py configs/<line>/exp-NNN.yml [--smoke]`：展開、驗證、記 commit/dirty、跑四 stage、寫 runs/。`resolved.yml` 內嵌所有被引用 exp 的展開結果與解析出的目錄名。`--smoke` 輸出到 `cache/_smoke/`，不記錄。stdout 只印 run 路徑，其餘走 stderr。
- 有 run 的 config 不可改，改參數 = 新 exp；不允許命令列覆蓋參數。
- stale = `stages.json.src_hashes` 與現在檔案 hash 不同。不是錯誤，summarize 標示，重跑即消失。
- `python summarize.py [--line] [--remote]`：從 runs/ 與 cache/（或 rclone cat 遠端）列表格，含各軸 name、指標、stale、commit；印每個變體被引用次數，0 為刪除候選。不讀任何 md。

## Research line

一個 line 只變動一個軸，`goal.md` 寫清楚。concluded 後不再加 exp，結論只引用 `runs/` 路徑，不手抄數字。任務不同（資料集不同）就是另一個 repo，不是另一個 line。

## 測試（pytest 全綠才算完成）

import 規則、config 可展開且 name 可解析、`runs/` 的 resolved.yml 與現在展開結果一致、cache key 對 src/params/up 任一變動都改變、被引用 exp 的 src/params 變動會改變引用者的 key、`init_from` 確實載入權重、`REQUIRES_REFS` 正反例、循環引用被拒、ACCEPTS 正反例、模型能力 Protocol 正反例、train `meta.json` 可重建模型、evaluate 拒絕非有限指標並寫出 canonical `metrics.json`、`tests/smoke.yml` 以 `--smoke` 在 CPU 60 秒內跑通、每個 goal.md 四段齊全。

## 禁止

Hydra/Lightning/W&B/MLflow/DVC；未記錄於變更記錄的新依賴；絕對路徑、機器名、密鑰進程式碼或 config；`pipeline/` 外的任務邏輯；手動編輯 `cache/`、`runs/`；為「以後可能需要」預留抽象層。

## 變更記錄（append-only，只寫重大改動）

格式：`YYYY-MM-DD — 改了什麼 — 為什麼 — 否決的替代方案`。新增軸、改介面簽名、改 cache key、遞增 CACHE_VERSION、新增依賴都算重大。

- 2026-08-27 — 新增跨實驗引用：`refs` 與 `train.init_from`（值為 `<line>/exp-NNN`），`up` 納入解析出的目錄名，`preprocess.run`/`algorithm.train` 增 `refs` 參數，`evaluate.score` 增 `out` — fine-tune、蒸餾 teacher、pseudo-label／特徵萃取、預測匯出在線性 stage 鏈下無法表達 — 否決的替代方案：直接引用 cache 目錄名（不可稽核、無法從 config 重算）；手動把 A 的 ckpt 當 source 資料擺入（脫離快取失效與 provenance）。
