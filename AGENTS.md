# AGENTS.md

架構、介面、快取協定、config 規則、測試清單全部在 `STRUCTURE.md`。動手前先讀完它；它是契約，程式碼與它衝突以它為準。

## 規則

- 任何重大改動（新增軸、改介面、改 cache key、遞增 CACHE_VERSION、新增依賴）都要先改 `STRUCTURE.md`，並在其「變更記錄」加一則：日期、改了什麼、為什麼、否決的替代方案。沒記錄的改動視為未完成。
- 新行為 = 新檔案；修 bug = 改檔並重跑所有 stale 的 run。
- 有 run 的 config 不可改；改參數就開新 exp。
- 結論只引用 `runs/` 路徑，不手抄數字進任何 md。
- 介面不清楚就查 `STRUCTURE.md`；沒寫就問，不要猜，不要為「以後可能需要」加抽象層。

## 工作順序

`pytest` 綠 → `run.py --smoke` 通 → `run.py` → `summarize.py` 確認 run 出現且無非預期 stale → 有結論才寫 `goal.md`。

## 完成的定義

pytest 全綠、run 出現在 summarize、沒有你造成的 stale 未處理、沒有未被任何 config 引用的新變體、重大改動已記錄。
