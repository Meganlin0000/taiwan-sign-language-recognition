#!/usr/bin/env python3
"""把一支「連續比很多次」的長影片，自動切成一支一支的片段。

為什麼要這個：
  用手機錄 38 次「起床」，比按 38 次開始／停止快得多，畫質也比筆電鏡頭好。
  錄完丟進來，它會找出每一次動作的起訖，自動切開並命名。

和原本前處理的差別：
  2025 原專題的前處理程式（未公開）固定只取影片前 70% 的影格，實測很多片段在被切掉時
  手還在動，動作尾端根本沒進訓練資料。這支改成依「手真的在動」的起訖切，
  並在前後各留一點緩衝，把完整動作留下來。

檔名會帶上詞、示範者代碼、日期與批次——這正是原本 4,232 支影片最缺的中繼資料
（它們全叫 clip_N.mp4，沒有任何來源資訊，後來只能用檔案修改時間推回錄影時段，有幾個詞還是切不乾淨）。

用法：
    python tools/split_takes.py 影片.mp4 --word 起床 --signer A
    python tools/split_takes.py 影片.mp4 --word 起床 --signer A --dry-run   # 只看會切幾段，不輸出

切完會產生一張預覽拼圖（每段的第一格拼在一起），先看那張圖確認切得對，
再決定要不要用。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data/片段"
# 預覽圖與中繼資料集中放，訓練影片資料夾裡只留 .mp4——
# 混著 jpg 與 csv 除了看起來亂，之後任何掃資料夾的程式也多一個出錯的機會。
# 預覽圖會拍到人臉，跟影片一樣放在 data/ 底下（.gitignore 已排除），不要上傳。
PREVIEW_DIR = ROOT / "data/切片預覽"
MANIFEST = ROOT / "data/切片紀錄.csv"


def parse_args():
    p = argparse.ArgumentParser(description="把連續錄製的長影片切成一次一段")
    p.add_argument("video", type=Path, help="要切的長影片")
    p.add_argument("--word", required=True, help="這支影片在比哪一個詞")
    p.add_argument("--signer", required=True, help="示範者代碼，例如 A、B、C（不要用真名）")
    p.add_argument("--out", type=Path, default=None, help="輸出資料夾（預設：data/片段/<詞>）")
    p.add_argument("--batch", default=None, help="批次代號，預設用現在的日期時間（到秒，跟錄影工具一樣）")
    p.add_argument("--min-take", type=float, default=0.6, help="一段動作最短幾秒（預設 0.6）")
    p.add_argument("--min-gap", type=float, default=0.7, help="兩段之間至少要靜止幾秒（預設 0.7）")
    p.add_argument("--pad-before", type=float, default=1.2,
                   help="每段起點之前多留幾秒（預設 1.2）；舉手的過程偵測不到手，靠留白補回來")
    p.add_argument("--pad-after", type=float, default=0.3, help="每段「終點之後」多留幾秒（預設 0.3）")
    p.add_argument("--expect", type=int, default=None, help="你比了幾次；填了會幫你對數量")
    p.add_argument("--sensitivity", type=float, default=0.85,
                   help="門檻靈敏度。切太少就調小（例如 0.7），切太多就調大（例如 1.4）")
    p.add_argument("--dry-run", action="store_true", help="只分析與畫圖，不輸出影片")
    return p.parse_args()


def hand_motion(video: Path, complexity: int = 1, stride: int = 2, width: int = 640):
    """逐幀算「手的移動量」，回傳 (每幀動量, fps, 總幀數, 尺寸)。

    用 MediaPipe 抓雙手關鍵點；沒偵測到手的幀動量記為 0，
    這樣「手放下休息」自然會變成分段的依據。
    """
    import mediapipe as mp

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"打不開影片：{video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"影片：{w}x{h}  {fps:.1f} fps  {total} 幀  約 {total/fps:.0f} 秒")

    # 速度取捨（實測結論，2026-09-22）：
    #   每 2 幀偵測一次、中間內插 → 26.5 → 11 毫秒/幀，切段結果不變（5/5）
    #   縮到 480 寬 → 會漏段（4/5），不採用
    #   輕量模型 complexity=0 單獨用沒事，但和跳幀合用會壞（2/5），不採用
    # 這支只用於「找動作起訖」，真正的訓練關鍵點由 抽關鍵點.py 另外擷取。
    STRIDE = stride

    motion, prev, present = [], None, []
    with mp.solutions.holistic.Holistic(min_detection_confidence=.5,
                                        min_tracking_confidence=.5,
                                        model_complexity=complexity) as holistic:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % STRIDE:
                # 略過的幀先補上一格的值，稍後一起內插
                motion.append(np.nan)
                present.append(present[-1] if present else False)
                i += 1
                continue
            small = cv2.resize(frame, (width, int(width * frame.shape[0] / frame.shape[1])))
            res = holistic.process(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            pts = []
            for lm in (res.left_hand_landmarks, res.right_hand_landmarks):
                pts.append(np.array([[p.x, p.y] for p in lm.landmark]).ravel()
                           if lm else np.zeros(42))
            cur = np.concatenate(pts)
            has = bool(res.left_hand_landmarks or res.right_hand_landmarks)
            present.append(has)
            # 跨越 STRIDE 幀的位移要除回來，動量尺度才和逐幀一致
            motion.append(0.0 if (prev is None or not has)
                          else float(np.abs(cur - prev).mean()) / STRIDE)
            prev = cur if has else None
            i += 1
            if i % 300 == 0:
                print(f"  分析中 {i}/{total}", end="\r", flush=True)
    cap.release()

    m = np.array(motion, dtype=float)
    if np.isnan(m).any():                      # 把略過的幀內插補回
        idx = np.arange(len(m))
        good = ~np.isnan(m)
        m = np.interp(idx, idx[good], m[good]) if good.any() else np.zeros_like(m)
    print(f"  分析完成 {i} 幀{' ' * 20}")
    return m, np.array(present), fps, (w, h)


def find_takes(motion, present, fps, min_take, min_gap, pad_before, pad_after=None, sensitivity=1.0):
    """依動量找出每一次動作的起訖幀。

    先平滑再判斷，不要被瞬間中斷騙：MediaPipe 常常有零點幾秒抓不到手，
    如果直接拿逐幀動量去切，一次完整的動作會被那個空檔切成兩半，
    兩半又都太短而被丟掉——整次動作就這樣消失了（實測發生過）。
    """
    if not len(motion):
        return []

    # 1) 平滑：用約 0.3 秒的移動平均填掉偵測中斷造成的瞬間低谷
    win = max(3, int(round(fps * 0.3)) | 1)
    k = np.ones(win) / win
    smooth = np.convolve(motion, k, mode="same")

    act = smooth[smooth > 1e-6]
    if not len(act):
        return []

    # 2) 門檻：取「安靜」與「動作」兩個水準的中間，而不是固定比例。
    #    安靜水準用整段的低分位數（含完全沒動的幀），動作水準用高分位數。
    quiet = float(np.percentile(smooth, 40))
    loud = float(np.percentile(act, 75))
    thr = (quiet + loud) * 0.35 * sensitivity
    # 保險：避免整段都被當成動作。
    # 這裡一定要用高分位數而不是最大值——實測「老師」那支 254 秒的影片裡有一個
    # 異常大的動作尖峰（是中位數的 100 倍），用 max*0.04 會把門檻拉到正常動作的四倍，
    # 結果 254 秒只切出 2 段（同長度的「好」切出 48 段）。
    thr = max(thr, float(np.percentile(smooth, 99)) * 0.10)
    moving = smooth > thr

    # 3) 補洞：短於 min_gap 的中斷視為同一次動作
    gap_frames = max(1, int(min_gap * fps))
    filled = moving.copy()
    run = 0
    for i, v in enumerate(moving):
        if v:
            if 0 < run <= gap_frames:
                filled[i - run:i] = True
            run = 0
        else:
            run += 1

    takes, start = [], None
    for i, v in enumerate(filled):
        if v and start is None:
            start = i
        elif not v and start is not None:
            takes.append((start, i))
            start = None
    if start is not None:
        takes.append((start, len(filled)))

    # 前後留白不對稱：舉手進入動作的那段偵測不到手，動量是零，
    # 只能靠「起點往前多留」把它補回來；結束時手還在畫面上，不需要留那麼多。
    if pad_after is None:
        pad_after = pad_before
    pre, post = int(pad_before * fps), int(pad_after * fps)
    min_f = int(min_take * fps)
    out = []
    for a, b in takes:
        if b - a < min_f:
            continue
        out.append((max(0, a - pre), min(len(motion), b + post)))
    return out


def contact_sheet(video: Path, takes, fps, dest: Path):
    """每段抽一格拼成一張圖，讓人一眼看出切得對不對。"""
    cap = cv2.VideoCapture(str(video))
    thumbs = []
    for k, (a, b) in enumerate(takes):
        cap.set(cv2.CAP_PROP_POS_FRAMES, (a + b) // 2)
        ok, frame = cap.read()
        if not ok:
            continue
        t = cv2.resize(frame, (192, int(192 * frame.shape[0] / frame.shape[1])))
        cv2.putText(t, f"{k+1}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 0), 4)
        cv2.putText(t, f"{k+1}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2)
        thumbs.append(t)
    cap.release()
    if not thumbs:
        return None
    cols = min(8, len(thumbs))
    rows = (len(thumbs) + cols - 1) // cols
    th, tw = thumbs[0].shape[:2]
    sheet = np.full((rows * th, cols * tw, 3), 30, np.uint8)
    for k, t in enumerate(thumbs):
        r, c = divmod(k, cols)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    cv2.imwrite(str(dest), sheet)
    return dest


def cut(video: Path, a, b, fps, dest: Path):
    """用 ffmpeg 重新編碼切出片段。重新編碼比較慢但起訖會準。"""
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{a/fps:.3f}", "-i", str(video),
         "-t", f"{(b-a)/fps:.3f}", "-an", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-pix_fmt", "yuv420p", str(dest)],
        check=True)


def short_batch(batch: str) -> str:
    """20260922_121218 → 0922-121218。只省年份，秒數必須留著。

    教訓（2026-09-22 實際踩到）：一度把秒數也砍掉變成 0922-1218，
    結果同一分鐘內錄的三次全部撞名互相覆蓋，切片從 11 支變成 9 支。
    檔名短一點不值得拿唯一性去換。
    """
    try:
        d, t = batch.split("_")
        return f"{d[4:8]}-{t}"
    except Exception:
        return batch


def split_one(video: Path, word: str, signer: str, batch: str | None = None,
              out: Path | None = None, min_take=0.6, min_gap=0.7,
              pad_before=1.2, pad_after=0.3,
              expect: int | None = None, sensitivity=0.85, dry_run=False, quiet=False):
    """切一支影片。回傳 (段數, 預覽圖路徑, 輸出資料夾)。錄影工具與批次切片都呼叫這個。"""
    say = (lambda *a, **k: None) if quiet else print
    if any(c in signer for c in "_*?[]/\\"):
        raise SystemExit(f"示範者代碼不能有底線或 * ? [ ] / \\ 這些符號（檔名靠底線分段）：{signer}")
    if not video.exists():
        raise SystemExit(f"找不到影片：{video}")
    batch = batch or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out or (DEFAULT_OUT / word)
    out_dir.mkdir(parents=True, exist_ok=True)

    motion, present, fps, size = hand_motion(video)
    takes = find_takes(motion, present, fps, min_take, min_gap,
                       pad_before, pad_after, sensitivity)

    say(f"\n找到 {len(takes)} 段動作")
    if expect:
        d = len(takes) - expect
        if d == 0:
            say(f"  和你說的 {expect} 次相符")
        else:
            say(f"  你說比了 {expect} 次，但切出 {len(takes)} 段（差 {d:+d}）。"
                f"\n  切太少 → 加 --sensitivity 0.7（門檻放寬），或錄的時候每次之間停久一點。"
                f"\n  切太多 → 加 --sensitivity 1.4，或加大 --min-gap。")
    for k, (s_, e) in enumerate(takes[:5]):
        say(f"  第 {k+1} 段：{s_/fps:5.2f}s → {e/fps:5.2f}s（{(e-s_)/fps:.2f} 秒）")
    if len(takes) > 5:
        say(f"  …其餘 {len(takes)-5} 段")

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    sheet = contact_sheet(video, takes, fps, PREVIEW_DIR / f"{word}__{signer}__{batch}.jpg")
    if sheet:
        say(f"\n切片預覽圖：{sheet}\n  先打開這張看每一段有沒有抓對，再決定要不要用。")
    if dry_run:
        say("\n（--dry-run：沒有輸出影片）")
        return len(takes), sheet, out_dir

    # 檔名要短到 Finder 不會把關鍵部分截掉，但四個欄位（詞／示範者／批次／編號）
    # 一個都不能省——原本 4,232 支全叫 clip_N.mp4，就是因為沒有這些資訊，
    # 事後只能用檔案修改時間推回錄影時段，有幾個詞還是切不乾淨。完整批次與日期另存在 切片紀錄.csv。
    short = short_batch(batch)
    clash = [f"{word}_{signer}_{short}_{k+1:02d}.mp4" for k in range(len(takes))]
    exist = [n for n in clash if (out_dir / n).exists()]
    if exist:
        raise SystemExit(f"停下來：批次 {batch} 已經有 {len(exist)} 個同名檔案，"
                         f"再輸出會蓋掉它們。\n請加上 --batch 換一個代號，例如 --batch {batch}b")

    # 每一段的手部偵測率。這個數字是免費的——找動作起訖時已經逐幀跑過
    # MediaPipe，present 就是「這一幀有沒有抓到手」。以前沒印出來，
    # 結果是要等抽完關鍵點、再另外寫程式去掃，才知道某個詞拍壞了。
    # 偵測率很低的片段多半是手出了畫面或拍糊了，早點知道就能當場重拍。
    # 這只是提醒重拍，不是要刪片段：開發時試過照偵測率刪片段，刪多了反而比隨機刪還差。
    #
    # 只算「動作本體」，不算前後補的那段：pad_before 的 1.2 秒是手還在
    # 舉起來的過程，那時候 MediaPipe 本來就抓不到手（手在畫面外）。
    # 把補的部分算進去會讓同一批影片從 97% 掉到 57%，變成天天誤報。
    pre = int(pad_before * fps)
    post = int((pad_after if pad_after is not None else 0) * fps)
    rows = []
    rates = []
    for k, (s_, e) in enumerate(takes):
        name = f"{word}_{signer}_{short}_{k+1:02d}.mp4"
        cut(video, s_, e, fps, out_dir / name)
        a0, a1 = int(s_) + pre, max(int(s_) + pre + 1, int(e) - post)
        seg = present[a0:a1]
        if len(seg) < 3:                       # 補的部分太長就退回整段
            seg = present[int(s_):int(e)]
        rate = float(seg.mean()) if len(seg) else 0.0
        rates.append(rate)
        rows.append({"檔名": name, "詞": word, "示範者": signer, "批次": batch,
                     "錄製日期": batch.split("_")[0] if "_" in batch else "",
                     "來源影片": video.name, "起始秒": round(s_ / fps, 3),
                     "結束秒": round(e / fps, 3), "長度秒": round((e - s_) / fps, 3),
                     "偵測率": round(rate, 3)})
    if rows:
        meta = MANIFEST
        cols = list(rows[0].keys())
        if meta.exists():
            with meta.open(encoding="utf-8") as f:
                old_rows = list(csv.DictReader(f))
            old_cols = list(old_rows[0].keys()) if old_rows else []
            if old_cols and old_cols != cols:
                # 加了新欄位（例如偵測率）之後，舊檔的標題列會少一欄。
                # 直接附加會讓每一列的欄位數對不上，整份 csv 壞掉。
                # 這裡把整份重寫一次，舊列缺的欄位留空。
                merged = old_cols + [c for c in cols if c not in old_cols]
                with meta.open("w", newline="", encoding="utf-8") as f:
                    wr = csv.DictWriter(f, fieldnames=merged)
                    wr.writeheader()
                    for r in old_rows:
                        wr.writerow({c: r.get(c, "") for c in merged})
                cols = merged
        new = not meta.exists()
        with meta.open("a", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            if new:
                wr.writeheader()
            wr.writerows([{c: r.get(c, "") for c in cols} for r in rows])
        say(f"\n完成：{len(rows)} 支片段輸出到 {out_dir}")

    if rates:
        import numpy as _np
        avg = float(_np.mean(rates))
        bad = [i + 1 for i, r in enumerate(rates) if r < 0.70]
        mark = "✅" if avg >= 0.90 else ("⚠️" if avg >= 0.80 else "❌")
        say(f"\n{mark} 手部偵測率：平均 {avg*100:.0f}%"
            f"（{'良好' if avg >= 0.90 else ('偏低' if avg >= 0.80 else '太低')}）")
        if bad:
            say(f"   偏低的片段（不到 70%）：第 {'、'.join(map(str, bad[:12]))} 段"
                + (f" 等 {len(bad)} 段" if len(bad) > 12 else f"　共 {len(bad)} 段"))
        if avg < 0.90:
            say("   建議：光線再亮一點、動作放慢一點、手不要超出畫面上緣，"
                "然後這個詞重拍一次。")
    return len(takes), sheet, out_dir


def main():
    a = parse_args()
    split_one(a.video, a.word, a.signer, a.batch, a.out, a.min_take, a.min_gap,
              a.pad_before, a.pad_after, a.expect, a.sensitivity, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
