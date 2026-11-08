#!/usr/bin/env python3
"""錄影工具：一個詞按一次，連續比很多次，自動切成一支一支。

為什麼另寫一支，不沿用 2025 原專題的錄影程式（未公開）：
  一、原本那支把 `fps = int(capture.get(CAP_PROP_FPS))` 直接餵給 VideoWriter。
      macOS 的內建鏡頭常常回傳 0，fps=0 寫出來的 mp4 是壞的——可能錄兩小時才發現。
      這裡改成實際量測影格率。
  二、原本那支是「比一次按一次」。要錄 38 次就得按 76 下。這裡改成錄一整段再自動切。
  三、原本那支檔名是 clip_N.mp4，沒有示範者與批次資訊。整個舊資料集就是因為這樣，
      事後只能用檔案修改時間推回錄影時段，有幾個詞還是切不乾淨。這裡的檔名一律帶上詞、示範者、日期、批次。

錄好的原片存在 data/錄影原片/，按停止後自動呼叫 split_takes.py 切成一支一支，
放到 data/片段/{詞}/；接著用 抽關鍵點.py 轉成關鍵點。需要系統裝好 ffmpeg（含 ffprobe）。
公開版拿掉了原本「錄網站示範影片」與「錄完先不切片」兩個選項（示範影片沒有公開）。

啟動：
    python tools/錄影工具.py          然後用瀏覽器開 http://127.0.0.1:5003
第一次執行時 macOS 會問要不要讓終端機使用鏡頭，要按允許。
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data/錄影原片"
TAKES_DIR = ROOT / "data/片段"
SPLITTER = Path(__file__).resolve().parent / "split_takes.py"
# 下拉選單的詞：預設是網站模型認得的 25 個詞。要錄別的詞，把 VOCAB 指到你自己的詞表
# （JSON 陣列，例如 ["早安", "謝謝"]）；不要直接改 web/model/labels.json，那個檔要跟模型一致。
VOCAB = ROOT / "web/model/labels.json"


def build_word_groups():
    """下拉選單的詞，照詞表的順序。"""
    if not VOCAB.exists():
        return []
    d = json.loads(VOCAB.read_text(encoding="utf-8"))
    words = [d[str(i)] for i in range(len(d))] if isinstance(d, dict) else list(d)
    return [{"名稱": "要錄的詞", "項目": [{"值": w, "顯示": w} for w in words]}]


def clip_counts():
    """每個詞目前已經有幾支片段，讓錄的人知道進度。"""
    counts = {}
    if TAKES_DIR.exists():
        for d in TAKES_DIR.iterdir():
            if d.is_dir():
                counts[d.name] = len(list(d.glob("*.mp4")))
    return counts

parser = argparse.ArgumentParser(description="錄影工具；只有一般啟動才會開鏡頭")
parser.add_argument("--port", type=int, default=5003)
parser.add_argument("--check", action="store_true", help="只檢查環境，不開鏡頭")
args = parser.parse_args()

if args.check:
    ok = {"ffmpeg": shutil.which("ffmpeg") is not None,
          "ffprobe": shutil.which("ffprobe") is not None,
          "切片器存在": SPLITTER.exists()}
    for m in ("cv2", "flask", "mediapipe"):
        try:
            __import__(m); ok[m] = True
        except Exception as e:
            ok[m] = f"缺少：{e}"
    print(json.dumps(ok, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all(v is True for v in ok.values()) else 1)

import cv2  # noqa: E402
from flask import Flask, Response, jsonify, request  # noqa: E402

app = Flask(__name__)

cap = None
lock = threading.Lock()
_g = build_word_groups()
# 預設選詞表的第一個詞，跟下拉選單打開時看到的一致
FIRST_WORD = _g[0]["項目"][0]["值"] if _g else "早安"

state = {"recording": False, "word": FIRST_WORD, "signer": "", "frames": 0,
         "started": 0.0, "fps": 30.0, "writer": None, "path": None, "message": ""}


def measure_fps(capture, n=25):
    """實際量影格率。不能相信 CAP_PROP_FPS，macOS 常常回 0。"""
    for _ in range(5):
        capture.read()
    t0 = time.time()
    got = 0
    for _ in range(n):
        if capture.read()[0]:
            got += 1
    dtsec = time.time() - t0
    fps = got / dtsec if dtsec > 0 else 0
    reported = capture.get(cv2.CAP_PROP_FPS) or 0
    chosen = fps if 5 <= fps <= 60 else (reported if 5 <= reported <= 60 else 30.0)
    print(f"影格率：實測 {fps:.1f}、鏡頭回報 {reported:.1f} → 採用 {chosen:.1f}")
    return float(chosen)


def open_camera():
    global cap
    c = cv2.VideoCapture(0)
    if not c.isOpened():
        raise SystemExit("打不開鏡頭。若 macOS 沒有跳出權限詢問，"
                         "請到「系統設定 → 隱私權與安全性 → 相機」允許終端機。")
    # 盡量要到 720p，比舊資料的 640x480 清楚
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    state["fps"] = measure_fps(c)
    w = int(c.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(c.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"鏡頭已開啟：{w}x{h}")
    cap = c
    return c


PAGE = """<!doctype html><html lang="zh-Hant-TW"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>錄影工具</title>
<style>
:root{font-family:-apple-system,"PingFang TC","Microsoft JhengHei",sans-serif;font-size:17px}
*{box-sizing:border-box}
body{margin:0;background:#0E1013;color:#F4F6F8;line-height:1.6}
/* 整頁固定成一個螢幕高：鏡頭吃掉剩餘空間，按鈕永遠在畫面內。
   固定鏡頭高度在小筆電上會把按鈕擠出畫面，在大螢幕又浪費空間，實測過兩端都不行。 */
.wrap{max-width:1100px;margin:0 auto;padding:12px 20px 14px;
 height:100vh;height:100dvh;display:flex;flex-direction:column;gap:0}
h1{font-size:1.1rem;margin:0 0 2px;flex:0 0 auto}
.sub{color:#98A2AE;font-size:.82rem;margin:0 0 10px;flex:0 0 auto}
/* 限高，讓鏡頭、選單、按鈕全部塞進一個螢幕——錄的人要坐遠比手語，不能讓按鈕跑到畫面外 */
.stage{position:relative;background:#07090B;border-radius:14px;overflow:hidden;
 flex:1 1 auto;min-height:240px}
.stage img{width:100%;height:100%;object-fit:contain;display:block;transform:scaleX(-1)}
.dot{position:absolute;top:14px;left:14px;display:flex;align-items:center;gap:10px;
 padding:10px 18px;border-radius:999px;background:rgba(8,10,13,.82);font-size:1.05rem;
 font-weight:700}
.stage.on{outline:4px solid #FF6B6B;outline-offset:-4px}
.dot b{width:14px;height:14px;border-radius:50%;background:#566174;display:inline-block}
.dot.on b{background:#FF6B6B;animation:p 1s infinite}
@keyframes p{50%{opacity:.25}}
.row{display:flex;gap:12px;flex-wrap:wrap;align-items:flex-end;margin-top:10px;flex:0 0 auto}
label{display:block;font-size:.8125rem;color:#98A2AE;margin-bottom:6px}
select,input{width:100%;min-height:42px;padding:8px 12px;background:#0E1013;color:#F4F6F8;
 border:1px solid #566174;border-radius:10px;font:inherit}
.f{flex:1 1 180px;min-width:0}
select{max-width:100%}
button{min-height:50px;padding:10px 24px;border:1px solid transparent;border-radius:10px;
 font:inherit;font-weight:700;cursor:pointer}
.go{background:#5AA9FF;color:#06121F;flex:1 1 220px}
.stop{background:#FF6B6B;color:#1A0606;flex:1 1 220px}
.discard{background:transparent;border-color:#566174;color:#98A2AE;flex:0 0 auto;font-weight:400}
.discard:hover:not(:disabled){border-color:#FF6B6B;color:#FF6B6B}
button:disabled{opacity:.45;cursor:not-allowed}
.msg{margin-top:10px;padding:10px 14px;border-radius:10px;background:#171A1F;
 font-size:.85rem;white-space:pre-wrap;flex:0 1 auto;overflow:auto;min-height:3em}
.tip{margin-top:8px;color:#98A2AE;font-size:.78rem;flex:0 0 auto}
.tip summary{cursor:pointer;padding:6px 0;min-height:32px}
.tip div{padding-top:4px;line-height:1.7}
.tip b{color:#F4F6F8}
</style></head><body><div class="wrap">
<h1>錄影工具</h1>
<p class="sub">選一個詞，按一次開始，連續比很多次，按停止。它會自動切成一支一支。</p>
<div class="stage"><img src="/stream"><div class="dot" id="dot"><b></b><span id="dotText">未錄影</span></div></div>
<div class="row">
 <div class="f"><label for="signer">示範者代碼（自己取，例如 A、B、C；不要用真名）</label>
  <input id="signer" placeholder="A"></div>
 <div class="f"><label for="q">搜尋詞彙（可不填）　<span id="qcount"></span></label>
  <input id="q" placeholder="要錄的詞已在選單最上面，這裡不用填"></div>
</div>
<div class="row">
 <div class="f"><label for="word">要比的詞　<span id="wcount"></span>　←　從這裡選</label>
  <select id="word" size="1"></select></div>
</div>
<div class="row">
 <button class="go" id="go">開始錄影</button>
 <button class="stop" id="stop" disabled>停止</button>
 <button class="discard" id="discard" disabled>丟掉剛剛這支</button>
</div>
<div class="msg" id="msg">先填示範者代碼，選一個詞，再按開始。</div>
<details class="tip"><summary>錄影要點（點開看）</summary><div>
<b>怎麼比才有用：</b>每一次之間讓雙手完全放下、停約一秒，切片才切得乾淨。<br>
<b>要刻意變化：</b>每錄十幾次就換一次站位（站左邊、右邊、前後移動、椅子調高低），讓同一個詞的錄影不要都長得一樣。<br>
<b>另一隻手：</b>訓練時會自動加左右鏡像，但那只是把關鍵點左右翻過來模擬換手，真人用另一隻手比準不準沒有量過；有人會用另一隻手比的話，兩隻手都錄一些。<br>
<b>按空白鍵</b>就能開始／停止，坐遠也不用搆滑鼠。<br><b>切完先看預覽圖：</b>它會告訴你檔案位置，打開那張圖確認每一段都抓對了。
</div></details>
</div>
<script>
const $=i=>document.getElementById(i);
const GROUPS=__GROUPS__;
let COUNTS={};

// 選單依分組列出詞，再給一個搜尋框。
function render(filter=''){
  const sel=$('word'), keep=sel.value;
  sel.replaceChildren();
  const f=filter.trim();
  let shown=0;
  for(const g of GROUPS){
    const items=f?g.項目.filter(it=>it.顯示.includes(f)):g.項目;
    if(!items.length) continue;
    const og=document.createElement('optgroup'); og.label=g.名稱;
    for(const it of items){
      const n=COUNTS[it.值]||0;
      og.append(new Option(n?`${it.顯示}　已有 ${n} 支`:it.顯示, it.值));
      shown++;
    }
    sel.append(og);
  }
  if(!shown){ sel.append(new Option('找不到符合的詞','')); }
  if(keep&&[...sel.options].some(o=>o.value===keep)) sel.value=keep;
  const total=new Set(GROUPS.flatMap(g=>g.項目.map(i=>i.值))).size;
  $('wcount').textContent=f?`（篩出 ${shown} 個）`:`（共 ${total} 個）`;
  $('qcount').textContent=f?`　篩出 ${shown} 個`:`　只有要在 ${total} 個詞裡找特定的詞時才需要`;
}
async function refreshCounts(){
  try{ COUNTS=await fetch('/counts').then(r=>r.json()); }catch{}
  render($('q').value);
}
$('q').addEventListener('input',()=>render($('q').value));
// 空白鍵切換開始／停止；在輸入框或選單裡按空白鍵不攔截
$('discard').onclick=async()=>{
 $('discard').disabled=true;
 const r=await fetch('/discard',{method:'POST'}).then(r=>r.json());
 $('msg').textContent=r.message; refreshCounts();
};
document.addEventListener('keydown',e=>{
  if(e.code!=='Space') return;
  const t=e.target.tagName;
  if(t==='INPUT'||t==='SELECT'||t==='TEXTAREA') return;
  e.preventDefault();
  (!$('go').disabled ? $('go') : $('stop')).click();
});
refreshCounts();
try{$('signer').value=localStorage.getItem('signer')||''}catch{}
$('signer').addEventListener('input',()=>{try{localStorage.setItem('signer',$('signer').value)}catch{}});
function busy(on){$('go').disabled=on;$('stop').disabled=!on;
 if(on)$('discard').disabled=true;
 $('dot').classList.toggle('on',on);
 document.querySelector('.stage').classList.toggle('on',on);
 $('dotText').textContent=on?'錄影中':'未錄影';}
$('go').onclick=async()=>{
 const signer=$('signer').value.trim();
 if(!signer){$('msg').textContent='請先填示範者代碼。';return;}
 $('msg').textContent='開始錄影。連續比，每次之間把手放下停一秒。';
 busy(true);
 const r=await fetch('/start',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({word:$('word').value,signer})}).then(r=>r.json());
 if(r.status!=='ok'){$('msg').textContent=r.message;busy(false);}
};
$('stop').onclick=async()=>{
 $('stop').disabled=true;
 $('msg').textContent='停止錄影，正在分析動作並切片，請稍候…';
 const r=await fetch('/stop',{method:'POST'}).then(r=>r.json());
 $('msg').textContent=r.message; busy(false); refreshCounts();
 $('discard').disabled = r.status!=='ok';
};
</script></body></html>"""


@app.route("/")
def index():
    return PAGE.replace("__GROUPS__", json.dumps(build_word_groups(), ensure_ascii=False))


@app.route("/counts")
def counts():
    return jsonify(clip_counts())


# 抓影格與寫檔放在背景執行緒，不依賴瀏覽器有沒有在拉預覽串流。
#
# 原本兩件事寫在同一個產生器裡：預覽的 /stream 每跑一圈才 cap.read() 一次，
# 順便把影格寫進錄影檔。後果是——只要那條串流斷了（分頁切到背景、
# 重開工具後沒接回來、網路層被作業系統暫停），就完全不會呼叫 cap.read()，
# 於是「錄影狀態顯示正常、檔案一幀都沒有」，停止時得到 0 KB 的壞檔。
# 實際發生過兩次，中間隔兩分鐘，使用者以為是自己按錯。
_latest = {"jpeg": None}


def _grab_loop():
    # 等鏡頭真的開好。這個執行緒在模組載入時就可能被啟動，
    # 而 open_camera() 是在 main 裡才呼叫的——不等的話
    # 第一次 cap.read() 就會 AttributeError，執行緒靜靜死掉，
    # 之後錄影全部產生 0 KB 的空檔，而且畫面上看不出異常。
    while cap is None:
        time.sleep(.05)
    while True:
        with lock:
            if cap is None:
                break
            ok, frame = cap.read()
            if ok and state["recording"] and state["writer"] is not None:
                state["writer"].write(frame)
                state["frames"] += 1
        if not ok:
            time.sleep(.02)
            continue
        ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok2:
            _latest["jpeg"] = buf.tobytes()


def frames():
    """只負責把最新一張畫面送給瀏覽器。斷線不影響錄影。"""
    while True:
        j = _latest["jpeg"]
        if j is None:
            time.sleep(.03)
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
        time.sleep(1 / 30)


@app.route("/stream")
def stream():
    return Response(frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/start", methods=["POST"])
def start():
    d = request.get_json(silent=True) or {}
    word, signer = d.get("word"), (d.get("signer") or "").strip()
    if state["recording"]:
        return jsonify(status="error", message="已經在錄影了。")
    if not word or not signer:
        return jsonify(status="error", message="詞與示範者代碼都要填。")
    if any(c in signer for c in "_*?[]/\\"):
        return jsonify(status="error", message="示範者代碼不能有底線或 * ? [ ] / \\ 這些符號（檔名靠底線分段），建議用 A、B、C。")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RAW_DIR / f"{word}__{signer}__{stamp}.mp4"
    state["batch"] = stamp
    with lock:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 state["fps"], (w, h))
        if not writer.isOpened():
            return jsonify(status="error", message="無法建立影片檔，請確認磁碟空間。")
        state.update(recording=True, word=word, signer=signer, frames=0,
                     started=time.time(), writer=writer, path=path, last_path=None)
    print(f"開始錄影 → {path.name}")
    return jsonify(status="ok")


def _trash(p: Path) -> str:
    """移到垃圾桶而不是永久刪除，按錯了還救得回來。回傳放到哪裡；失敗時丟出例外。"""
    try:
        subprocess.run(["osascript", "-e",
                        f'tell application "Finder" to delete POSIX file "{p}"'],
                       check=True, capture_output=True, timeout=15)
        return "垃圾桶"
    except Exception:
        # 叫不動 Finder（例如不是 macOS）：有 ~/.Trash 就搬進去，沒有就搬到 data/已丟棄/，一樣救得回來
        trash, where = Path.home() / ".Trash", "垃圾桶"
        if not trash.is_dir():
            trash, where = ROOT / "data/已丟棄", "data/已丟棄/"
        trash.mkdir(parents=True, exist_ok=True)
        shutil.move(str(p), str(trash / p.name))
        return where


@app.route("/discard", methods=["POST"])
def discard():
    """丟掉剛剛錄的那一支，連同按停止時已經自動切好的片段與預覽圖。
    只能丟最後一支，避免誤刪之前的素材。"""
    path = state.get("last_path")
    if not path or not Path(path).exists():
        return jsonify(status="error", message="沒有可以丟掉的錄影。")
    p = Path(path)
    # 按停止就會自動切片：只丟原片的話，錄壞的片段會留在 data/片段/ 裡、之後被拿去訓練
    word, signer, batch = state.get("last_take") or (state["word"], state["signer"], state.get("batch", ""))
    d, _, t = batch.partition("_")
    short = f"{d[4:8]}-{t}" if t else batch          # 跟 split_takes.short_batch 相同
    clips = sorted((TAKES_DIR / word).glob(f"{glob.escape(word)}_{glob.escape(signer)}_{glob.escape(short)}_*.mp4"))
    preview = ROOT / "data/切片預覽" / f"{word}__{signer}__{batch}.jpg"
    where = "垃圾桶"
    try:
        for v in [*clips] + ([preview] if preview.exists() else []) + [p]:   # 原片最後丟，中途失敗還能再按一次
            where = _trash(v)
    except Exception as e:
        return jsonify(status="error", message=f"刪不掉：{e}")
    state["last_path"] = None
    return jsonify(status="ok", message=f"已把原片「{p.name}」和切出的 {len(clips)} 支片段丟到{where}。\n可以重錄這個詞了。")


@app.route("/stop", methods=["POST"])
def stop():
    if not state["recording"]:
        return jsonify(status="error", message="目前沒有在錄影。")
    with lock:
        state["recording"] = False
        state["writer"].release()
        state["writer"] = None
        path, n = state["path"], state["frames"]
        secs = time.time() - state["started"]
        state["last_path"] = str(path)
        state["last_take"] = (state["word"], state["signer"], state.get("batch", ""))

    # 檢查檔案是不是真的能播，不要錄完才發現是壞檔
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    dur = float(probe.stdout.strip() or 0)
    if dur <= 0:
        return jsonify(status="error",
                       message=f"影片檔有問題（長度 0 秒）：{path}\n請回報這個訊息，先不要繼續錄。")

    msg = [f"錄了 {secs:.0f} 秒、{n} 幀，原片存到：\n{path}", "", "正在切片…"]
    r = subprocess.run([sys.executable, str(SPLITTER), str(path),
                        "--word", state["word"], "--signer", state["signer"],
                        "--batch", state.get("batch", "")],
                       capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    for line in out.splitlines():
        if any(k in line for k in ("找到", "完成：", "切片預覽圖", "第 ", "中繼資料")):
            msg.append(line.strip())
    if r.returncode != 0:
        msg.append("\n切片失敗，但原片已經存起來了，之後可以手動再切。")
    return jsonify(status="ok", message="\n".join(msg))


if __name__ == "__main__":
    open_camera()
    # 鏡頭開好之後才啟動抓影格的執行緒
    threading.Thread(target=_grab_loop, daemon=True).start()
    print(f"\n錄影工具已啟動 → http://127.0.0.1:{args.port}")
    print("結束請回到這個視窗按 Control+C\n")
    try:
        app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)
    finally:
        if state.get("writer") is not None:
            state["writer"].release()
        if cap is not None:
            cap.release()
