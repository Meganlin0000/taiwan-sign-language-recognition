# 臺灣手語辨識

對著鏡頭比臺灣手語，瀏覽器即時辨識 25 個常用詞，轉成文字並唸出來。鏡頭畫面只在瀏覽器裡處理，不會上傳；只有用整句語音時，辨識出的詞（文字）會送到網站的語音服務，交給 Google 合成。

網站：<https://taiwan-sign-language.pages.dev>

## 怎麼運作

1. MediaPipe Holistic 從每一幀抓出身體 33 點、雙手各 21 點，共 75 個關鍵點。
2. 只留看得到手的那一段，重取樣成固定 80 幀。
3. 每幀 291 個數值：258 個關鍵點數值（身體 33 點 × 4、雙手 42 點 × 3，x、y 以雙肩中點為原點、肩寬為尺度），加上 33 個指向幾何特徵（每隻手 15 個：手指方向、手掌朝向、指尖相對鼻子和胸口的位置等；身體 3 個：肩膀連線方向、兩手食指尖距離）。
4. 三層 LSTM 讀完這 80 幀，輸出 25 個詞的機率。
5. 網站每 0.25 秒看一次最近 2.5 秒的畫面；最高機率 ≥ 0.90、而且連續 3 次都是同一個詞，才把它寫進句子（手放下的那一刻會用整段動作再判一次）。

想改成手機 App 的話，網站的前處理就是模型的輸入規格，以 `web/js/preprocess.js` 和 `training/tslkit.py` 為準：畫面先左右翻轉再送進 MediaPipe（`tools/抽關鍵點.py` 的 `cv2.flip`），258 欄的排列順序、換算成 16:9、重取樣成 80 幀、33 個指向幾何特徵，最後送進模型的是 [1, 80, 291]；推論時還有鏡像平均與先驗校正（`web/js/inference-worker.js`）。改用新版 MediaPipe（例如 HolisticLandmarker）的話，座標可能跟訓練資料不同，要重抽關鍵點、重新訓練，再用換人測試確認準確度。

## 資料夾

| 位置 | 內容 |
|---|---|
| `web/` | 網站本體（純靜態檔）。`js/` 是辨識流程，`model/` 是訓練好的模型，`vendor/` 是 TensorFlow.js 與 MediaPipe |
| `functions/` | 選用。Cloudflare Pages 的整句語音服務，要自己的 Google 語音合成金鑰 |
| `training/` | 訓練：`tslkit.py` 前處理、`pointing.py` 指向幾何特徵、`train.py` 訓練與評分、`rotate_test.py` 每個人輪流當考卷的換人測試 |
| `tools/` | 錄影、切片、抽關鍵點，以及把 Keras 模型轉成網站用的格式 |

只要網站的話，`web/`（加上選用的 `functions/`）就夠了。`training/`、`tools/` 和兩個 `requirements` 檔網站執行時用不到，是換詞、用自己的資料重訓模型時才需要的錄影、抽關鍵點、訓練、轉檔流程。

## 怎麼跑

需要 Python 3.11、ffmpeg（含 ffprobe，錄影與切片用）和一台有鏡頭的電腦。作者只在 Apple Silicon 的 macOS 上跑過：Intel Mac 裝不了 tensorflow 2.19.1，Windows 和 Linux 沒有實測（錄影工具也只在 macOS 用過）。指令都在這個資料夾的最上層執行；Windows 把 `.venv/bin/` 換成 `.venv\Scripts\`。沒有 Python 3.11 的話，macOS 可以用 `brew install python@3.11`（ffmpeg 用 `brew install ffmpeg`）；Windows 到 python.org 下載 3.11，第 2、6 步的 `python3.11` 改成 `py -3.11`。tensorflow 2.19.1 不支援 Python 3.13 以上。

1. 只看網站：`python3 -m http.server 8000 --directory web`，開 <http://localhost:8000>（鏡頭只能在 localhost 或 HTTPS 下開）
2. 建環境：`python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt`，裝好後跑 `.venv/bin/python tools/錄影工具.py --check`，全部是 true 就代表錄影、切片和抽關鍵點的環境好了（它不檢查第 5 步訓練用的 tensorflow）
3. 錄影與切片：`.venv/bin/python tools/錄影工具.py`，開 <http://127.0.0.1:5003>，選詞、填示範者代碼（A、B、C…），連續比很多次，按停止後自動切成一支一支，存到 `data/片段/{詞}/{詞}_{示範者}_{批次}_{編號}.mp4`。錄的時候：每一次之間雙手完全放下、停約一秒（自動切片靠「手放下」來分段）；拍上半身，雙肩都要入鏡，鏡頭盡量跟臉同高（仰拍容易認錯）；每錄十幾次換一次站位。錄影工具啟動時終端機會印出「鏡頭已開啟：寬x高」，可以確認實際解析度。自己用別的方式錄也可以：一段連續比很多次的長片，用 `.venv/bin/python tools/split_takes.py 影片 --word 詞 --signer 代碼` 切，輸出就是這個檔名；已經一次一支的，照這個檔名放好就行（用底線分成剛好四段，詞和示範者代碼裡不能有底線（示範者代碼建議只用英文字母或數字）；看不出示範者的檔名，`training/tslkit.py` 會直接報錯），副檔名要是小寫 `.mp4`（`抽關鍵點.py` 只找 `.mp4`，手機錄的 `.MOV`、`.MP4` 會被略過）。請用 16:9 橫拍：網站送進模型前會把畫面換算成 16:9（`web/js/preprocess.js` 的 `REF_ASPECT`），主要用別的比例錄的話，要把它改成那個比例
4. 抽關鍵點：`.venv/bin/python tools/抽關鍵點.py`，每支片段變成 `data/關鍵點/{詞}/` 底下一個 (幀數, 258) 的 `.npy`
5. 訓練：`.venv/bin/python training/train.py --data data/關鍵點 --train A B C --test D --out output/model`。一開始會列出每位示範者有幾支，先核對再往下跑。每位訓練示範者會各抽 12% 當驗證集，所以每個人自己的片段裡，每個詞至少要 2 支，總數也至少要 201 支（25 個詞時；每個詞一樣多的話就是每人每詞 9 支）。不夠時 `train.py` 會先停下來，說明差多少。考卷印出的三行分數裡，「網站做法（鏡像平均再除以先驗）」才是網站實際的算法；`考卷D_混淆矩陣.csv` 可以用 Excel 開，看哪些詞常被互相認錯。想讓每個人輪流當考卷：`.venv/bin/python training/rotate_test.py --data data/關鍵點 --signers A B C D --out output/rotate`，每一輪就是跑一次 `train.py`，所以每一位都要達到上面的支數；結果彙整在 `output/rotate/換人測試總表.json`
6. 轉檔（另一個環境）：`python3.11 -m venv .venv-convert && .venv-convert/bin/pip install --no-deps -r requirements-convert.txt`，再執行 `.venv-convert/bin/python tools/轉檔_keras轉tfjs.py output/model/model.keras output/tfjs`
7. 放進網站：把 `output/tfjs/` 的 `model.json` 與 `.bin`、`output/model/` 的 `labels.json` 與 `class-prior.json` 複製到 `web/model/`，回到第 1 步實測

### 換成自己的詞

- `web/guide.json`（示範選單）：每筆要有 `id`（labels.json 的編號）、`label`、`unit`、`unitTitle`、`seconds`（同一組裡排序用）；`sourceLabel`、`url` 是教材出處，程式沒有用到。選單只列 `unit` 是 1、2、3 的詞，要更多組就改 `web/js/app.js` 的 `for (const u of [1, 2, 3])`。
- `functions/api/tts.js` 的詞表。
- `tools/錄影工具.py` 的 `VOCAB`：錄影的下拉選單預設讀 `web/model/labels.json`，要錄新詞就把它指到你自己的詞表（JSON 陣列，例如 `["早安", "謝謝"]`）。
- 詞數不是 25 時，寫死的 25 都要改：`web/js/inference-worker.js` 第 28、54、60 行，`web/js/continuous-core.js` 第 168、457 行，`web/js/recognition-core.js` 第 25 行（`rankScores`，每次辨識都會檢查，漏改的話網站一個詞都認不出來）。行號是寫這份說明時的位置，程式改過就會跑掉，找不到時在 `web/js` 裡搜尋 `25`。介面與說明文字的「25 個詞・選一個看示範」「25 個常用詞」在 `web/index.html`、`web/about/index.html` 和 `web/js/app.js`。
- `web/js/continuous-core.js` 有一道只給「同學」的手形關卡（`setHandshapeGate`），門檻是照這 25 個詞量的。新詞表沒有「同學」時會自動關掉；還有「同學」的話要重量，或在 `web/js/app.js` 第 486 行改成傳 `null` 關掉。

### 聲音與示範影片

公開版沒有附示範影片和單詞語音檔：示範影片會拍到人，不公開；單詞語音檔是用 Google 語音合成（臺灣中文 WaveNet A／B）另外產生的，也沒有附上，可以用 `functions/api/tts.js` 同樣的兩個聲音自己產生。所以只開網站時，選詞不會播示範，自動唸出、單詞和「全部唸出」都不出聲，辨識功能完整。畫面不會報錯；瀏覽器主控台會出現 `voices/manifest.json` 和 `guide-local.json` 兩個 404，因為公開版沒附這兩個檔，屬於正常。

要接上自己的檔案，照下面放（網站讀不到這兩個 JSON 就當作沒有，不用改程式）：

- 單詞語音：音檔放在 `web/voices/female/` 與 `web/voices/male/`（兩個資料夾的檔名相同），再寫 `web/voices/manifest.json`，格式是 `{"words": {"早安": "12.mp3", …}, "mid": {"早安": "12m.mp3", …}}`。`words` 是單獨唸的版本；`mid` 是唸在句子中間的版本，可以不放（沒有就用單詞版）。見 `web/js/speech.js`。
- 示範影片：影片放在 `web/videos/教學/`，再寫 `web/guide-local.json`，格式是 `{"words": {"早安": "早安.mp4", …}}`。有內容之後，在右欄選單選一個詞就會播它的示範；疊在影片上的骨架 `web/videos/教學/骨架/{詞}.json` 可以不放。見 `web/js/app.js`。

整句語音（選用，需要 Node.js）：在最上層建 `.dev.vars` 寫入 `GOOGLE_TTS_KEY=你的金鑰`，再執行 `npx wrangler pages dev`。句子至少要 2 個詞才會合成；第一次按「全部唸出」時，Google 超過 1.5 秒才回應就會沒聲音，再按一次就會唸。

### 自己部署

`web/` 是純靜態檔，放到靜態網站服務就能跑，但要放在網域的根目錄：說明頁（about、licenses、404）和首頁「關於這個網站」視窗裡的連結用的是從根目錄開始的路徑（例如 `/pages.css`、`/about/`）。放在 GitHub Pages 專案網址（帳號.github.io/repo 名稱/）這種子路徑底下時，辨識主頁可以用，說明頁會沒有樣式，頁尾和「關於這個網站」視窗裡的連結會失效。`web/_headers` 只有 Cloudflare Pages 和 Netlify 看得懂。

部署成自己的網站時，要改這些地方：

- 網址：把 `web/index.html`、`web/about/index.html`、`web/licenses/index.html`、`web/404.html` 裡的 `taiwan-sign-language.pages.dev` 換成自己的網址。canonical、og:url、og:image 和結構化資料都寫死這個網址，不換的話，分享預覽和搜尋引擎會指回作者的網站。
- 示範影片頁的連結：頁尾的「詞彙示範」、首頁「關於這個網站」視窗裡的「詞彙示範」，以及使用說明頁「不會比？先看示範」那段，都連到作者網站的示範影片頁（`/w/`）。公開版沒有這一頁，可以拿掉。
- 作者與影片說明：頁尾、「關於這個網站」視窗、使用說明頁和結構化資料裡的作者署名，以及「影片為作者自錄」，要改成自己的情況。MIT 授權只要求保留 LICENSE 全文（著作權聲明與授權條款），網站上的作者請寫自己。
- 沒有放示範影片的話，`web/index.html` 描述裡的「每個詞都有示範影片」和使用說明頁「挑一個詞就會播，可以放慢、看骨架」也要改寫。
- `.gitignore` 擋掉了 `web/videos/`、`web/voices/`、`web/guide-local.json` 和音檔，示範影片與語音檔不會進 Git，所以從 Git 自動部署（GitHub Pages、連 Git 的 Cloudflare Pages／Netlify）的網站不會有示範和語音；有放的話，要從本機直接上傳，例如 `npx wrangler pages deploy`。
- 用 Cloudflare Pages 的話，把 `wrangler.jsonc` 的 `name` 改成自己的專案名稱，部署指令是 `npx wrangler pages deploy`。部署後要用整句語音，要另外設金鑰：`npx wrangler pages secret put GOOGLE_TTS_KEY --project-name 你的專案名稱`（或在 Cloudflare 後台 Pages 專案的設定裡加秘密變數）。`functions/api/tts.js` 沒有用量上限，而且只擋得住別的網站的網頁、擋不住直接用程式呼叫，所以在 Google Cloud 要把金鑰限定只能用 Text-to-Speech，並設配額上限與預算提醒。

### 重訓的結果

公開版不含任何人的影像、影片與關鍵點，所以別人沒辦法重訓出 `web/model/` 那顆模型。在作者電腦上（Apple M5、只用 CPU、TensorFlow 2.19.1、Keras 3.11.3）用原本的資料跑 `train.py` 再轉檔，得到的 `model.json` 與權重檔跟 `web/model/` 的完全相同。換電腦或套件版本，結果可能有些微差異。

## 最該注意的一件事：不要隨機切分

同一場錄影切出來的片段彼此很像，隨機切分會讓它們同時出現在訓練和測試裡，分數嚴重高估。原專題用隨機切分得到 96%；改成依錄影時間切分後，同一位示範者、同樣的前處理只剩 60.4%。要知道模型對新使用者準不準，就留一個完全沒參與訓練的人當考卷（`train.py --test`）；更完整的做法是每個人輪流當一次考卷（`training/rotate_test.py`），看平均和人與人之間的差距。作者用這個方法得到四人平均 77.3%，最低 62.3%、最高 89.0%，差距很大，表示模型還很依賴某些人的打法和拍攝條件（這三個數字是每一輪三個隨機種子的平均，直接取最高機率，沒有做網站的鏡像平均和先驗校正；`rotate_test.py` 預設只跑一個種子，而且用網站的算法計分，所以自己跑出來的數字會不一樣）。網站的模型用示範者 A、B、C 訓練，D 只當考卷；不過開發網站設定時看過 D 的結果，所以 D 不算完全乾淨的考卷。要真正知道新使用者準不準，考卷從頭到尾都不要拿來調整任何設定。`train.py` 另外切出的驗證集跟訓練資料來自同一批錄影，只用來決定何時停止訓練，不能當成績。

| 評估方式 | 正確率 | 條件 |
|---|---|---|
| 原專題：隨機切分 | 96% | 同一場錄影的相似片段同時進了訓練與測試，嚴重高估 |
| 依錄影時間切分 | 60.4% | 同一位示範者，早錄的訓練、晚錄的測試 |
| 同上，改善前處理後 | 80.7% | 同上條件，三次訓練平均 |
| 四人輪流當考卷 | 平均 77.3%（62.3%～89.0%） | 考卷那個人完全沒參與訓練，每輪三個隨機種子平均 |

四列的條件都不一樣，不是同一顆模型的連續變化，也都不是網站即時辨識的正確率。

## 限制

- 只認 25 個孤立詞（詞彙取自[臺灣手語教材資源網](https://jung-hsingchang.tw/twsl/movies-uni1.html)第一冊第 1～3 單元），一次一個詞，不支援連續手語和句子翻譯。
- 閉集合分類：模型沒有「這不是手語」這一類，亂比或詞表外的手語也會被判成其中一個詞。
- 資料只有 4 位示範者、共 7,167 支片段：A、B、C 共 6,824 支拿來訓練，D 的 343 支只當考卷。換成沒參與訓練的人、鏡頭仰拍時都容易認錯。
- 示範者 A 的 4,232 支舊資料是 4:3 畫面，而且是用原專題的程式抽關鍵點、每支只取前 70% 影格；其他人是 16:9、完整長度。所以「換人」的差距裡，混有錄影與前處理條件的差異，不全是人的差異（換人測試裡 A 最低，可能也跟這有關）。
- 示範與訓練影片都由示範者自行錄製，打法沒有經過手語老師確認。教材網的影片沒有拿來訓練。
- 程式註解裡的數字（取景門檻、實測次數等）來自作者開發時的內部測試紀錄，沒有另外公開。

## 作者與分工

林欣頤（中原大學智慧運算與大數據學士班畢業）。

這原本是 2025 年四人共同完成的大學畢業專題，作者負責辨識端：資料整理、模型訓練、結果驗證和即時辨識流程整合。2026 年畢業後，作者把它重新評估、擴充到 4 位示範者，並把原本只能在本機執行的系統做成公開網站。這裡的程式都是 2026 年的版本，2025 年原專題的程式沒有收錄；模型架構（三層 LSTM）、MediaPipe 關鍵點抽取的設定，以及示範者 A 4,532 支訓練片段中的 4,232 支（約占網站模型訓練資料 6,824 支的六成；另外 300 支是 2026 年補錄）則來自原專題。程式大部分由 AI 程式助理協助撰寫，作者負責決定要做什麼、要驗證什麼，並逐項核對結果。

## 授權

程式碼與 `web/model/` 的模型採 MIT 授權，見 [LICENSE](LICENSE)。`web/vendor/` 裡的 TensorFlow.js 4.22.0 與 MediaPipe Holistic 0.5.1675471629 由 Google 開發，依 Apache License 2.0 授權，全文在各自資料夾的 `LICENSE`。

可以自由使用、修改、做成自己的 App，只要保留 LICENSE 全文（著作權聲明與授權條款）；`web/vendor/` 的兩個套件要保留各自的 LICENSE。
