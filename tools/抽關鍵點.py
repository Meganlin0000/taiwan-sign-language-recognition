#!/usr/bin/env python3
"""把切好的片段轉成 258 維關鍵點序列（每支影片一個 .npy）。

輸入：data/片段/{詞}/{詞}_{示範者}_{批次}_{編號}.mp4（split_takes.py 切出來的就是這個格式）
輸出：data/關鍵點/{詞}/{同檔名}.npy，內容是 (幀數, 258) 的 float32 陣列

和 2025 原專題的前處理程式（未公開）有兩點不同：

一、原本只取每支影片的前 70% 影格。這裡一律抽完整長度——
    要截短隨時可以在 numpy 裡切，但砍掉的東西補不回來。
    （示範者 A 的舊資料是用原本的程式抽的，所以只有前 70%。）

二、原本從頭到尾只開一個 Holistic 實例處理所有影片。
    MediaPipe 的追蹤狀態會跨影片殘留，前一支影片的最後一幀會影響下一支的第一幀。
    這裡每支影片各開一個，彼此獨立。

其餘設定刻意跟原本一模一樣，新舊資料才對得起來：
    cv2.flip(frame, 1)                左右翻轉（所以「左手欄位」不等於解剖學上的左手）
    cv2.cvtColor(BGR2RGB)             色彩空間
    min_detection_confidence=0.5
    min_tracking_confidence=0.5
    258 維排列 = 姿態 33×(x,y,z,可見度) + 左手 21×(x,y,z) + 右手 21×(x,y,z)
    偵測不到的區塊補全零（不是補上一幀，也不是內插）

用法：
    python tools/抽關鍵點.py                    # 全部，跳過已經抽過的
    python tools/抽關鍵點.py --只看不做          # 先看會處理幾支
    python tools/抽關鍵點.py --輸入 我的影片 --輸出 我的關鍵點
"""
import argparse
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "data/片段"
DEFAULT_OUT = ROOT / "data/關鍵點"

POSE_N, HAND_N = 33, 21


def extract_one(path, holistic_cls, cv2, np):
    """回傳 (幀數, 258) 的陣列。偵測不到的區塊是全零。"""
    seq = []
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    with holistic_cls(min_detection_confidence=0.5, min_tracking_confidence=0.5) as h:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)                 # 跟訓練資料同方向
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False
            r = h.process(img)
            pose = (np.array([[p.x, p.y, p.z, p.visibility]
                              for p in r.pose_landmarks.landmark]).flatten()
                    if r.pose_landmarks else np.zeros(POSE_N * 4))
            lh = (np.array([[p.x, p.y, p.z]
                            for p in r.left_hand_landmarks.landmark]).flatten()
                  if r.left_hand_landmarks else np.zeros(HAND_N * 3))
            rh = (np.array([[p.x, p.y, p.z]
                            for p in r.right_hand_landmarks.landmark]).flatten()
                  if r.right_hand_landmarks else np.zeros(HAND_N * 3))
            seq.append(np.concatenate([pose, lh, rh]))
    cap.release()
    return np.array(seq, dtype=np.float32) if seq else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--輸入", type=Path, default=DEFAULT_IN,
                    help="影片資料夾，底下要分成 {詞}/xxx.mp4")
    ap.add_argument("--輸出", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--只看不做", action="store_true")
    a = ap.parse_args()
    IN_DIR, OUT_DIR = a.輸入, a.輸出
    print(f"來源 {IN_DIR}\n輸出 {OUT_DIR}")

    todo = []
    for vid in sorted(IN_DIR.glob("*/*.mp4")):
        out = OUT_DIR / vid.parent.name / (vid.stem + ".npy")
        if not out.exists():
            todo.append((vid, out))
    done = len(list(OUT_DIR.glob("*/*.npy")))
    print(f"已抽過 {done} 支，這次要抽 {len(todo)} 支")
    if a.只看不做 or not todo:
        by_word = {}
        for v, _ in todo:
            by_word[v.parent.name] = by_word.get(v.parent.name, 0) + 1
        for w, n in sorted(by_word.items(), key=lambda x: -x[1]):
            print(f"  {w:<10} {n}")
        return 0

    os.environ.setdefault("GLOG_minloglevel", "3")
    import cv2
    import numpy as np
    import mediapipe as mp
    holistic_cls = mp.solutions.holistic.Holistic

    t0 = time.time()
    bad = []
    for i, (vid, out) in enumerate(todo, 1):
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = extract_one(vid, holistic_cls, cv2, np)
        if arr is None or len(arr) == 0:
            bad.append(vid.name)
            continue
        np.save(out, arr)
        # 每 5 支印一次，進度回饋要密到讓人確信它還活著。
        if i % 5 == 0 or i == len(todo):
            el = time.time() - t0
            rate = i / max(el, .001)
            left = (len(todo) - i) / max(rate, .001)
            bar_w = 24
            done_w = int(bar_w * i / max(len(todo), 1))
            bar = "█" * done_w + "░" * (bar_w - done_w)
            print(f"  {bar} {i}/{len(todo)}　{vid.parent.name}　"
                  f"{rate*60:.0f} 支/分　剩 {left/60:.1f} 分", flush=True)
    print(f"\n完成。失敗 {len(bad)} 支" + ("：" + "、".join(bad) if bad else ""))
    print(f"總關鍵點檔：{len(list(OUT_DIR.glob('*/*.npy')))} 個")
    return 0


if __name__ == "__main__":
    sys.exit(main())
