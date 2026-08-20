# Claude Code 透過轉接層讀取同一份契約

**Commit：** `8594550`

## 問題

這個 repo 的 agent 治理是為 Codex 建的：`AGENTS.md` 定義完整契約，`.agents/skills/` 下三個 skill
各自帶 `SKILL.md`、`references/` 與 Codex 專用的 `agents/openai.yaml`。Claude Code 兩者都讀不到——
它讀 `CLAUDE.md`，skill 只從 `.claude/skills/` 載入，而且沒有任何設定可以改變搜尋路徑
（`additionalDirectories` 只給檔案讀取權，不會載入 skill）。

結果是在這個 repo 裡開 Claude Code，它對所有治理規則一無所知：不知道實驗結果記錄後 config 就
immutable、不知道要先 dry-run 再 `--apply`、不知道支援聲明需要證據。這不是缺一個功能，是缺了
整套約束。

## 為什麼不用 symlink

最直覺的做法是把 `.claude/skills/` symlink 到 `.agents/skills/`，一份檔案兩邊共用。這條路被
repo 自己堵死了：`.agents/skills/training-manager/scripts/_common.py` 的 `reject_symlinks()` 會掃描
整個工作區（只跳過 `.git`、`.venv`、`datasets`、`checkpoints`、`runs`），一旦發現任何 symlink 就
中止初始化。加了 symlink 等於讓模板的初始化流程再也跑不起來。

`CLAUDE.md` 用 `@AGENTS.md` 匯入是官方文件給的另一個選項，而且官方本來就建議用它取代 symlink
（Windows 上建 symlink 需要管理員權限）。相對路徑基準是含 `@` 的檔案所在目錄，所以根目錄的
`CLAUDE.md` 寫 `@AGENTS.md` 正好匯入同目錄那份，一層就到位。

## 為什麼是轉接層而不是複製

skill 沒有 `@` 匯入這種機制，所以三個 `.claude/skills/*/SKILL.md` 必須是實體檔案。剩下的選擇是
複製內容還是只放指標。

複製必然漂移。這個 repo 的歷史已經證明了 skill 是會一起變的：`d00d973` 那次為了讓 progress
規則進入每個流程，三個 `SKILL.md` 連同三份 references 全部被改。若當時 `.claude/` 已存在且是
複製版，那次改動會留下六個檔案裡的三份過期副本，而且沒有任何檢查抓得到——CI 的兩支 validator
都只掃 `.agents/skills/`。

所以轉接層只保留 frontmatter 與一句「權威來源在 `.agents/`，動手前先讀它」。frontmatter 的
`description` 逐字沿用 `.agents/` 那份，因為那段文字同時是 Claude 判斷何時自動觸發的依據，
兩邊逐字相同才能讓觸發條件一致。代價是 Claude 必須多讀一個檔案才知道細則，換來的是規則永遠
只有一份。

## 後續修正

首次實作時，每個轉接檔還額外寫了三條摘要規則（共九條），想讓 Claude 在還沒讀權威檔前就不會犯錯。
那是同一個錯誤的縮小版：那些句子是改寫過的 paraphrase，沒有檢查抓得到漂移，其中一條還逐字列出
四個 manifest 檔名。後來全部移除，讓轉接層真正只做轉接。
