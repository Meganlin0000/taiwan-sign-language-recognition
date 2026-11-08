"""臺灣手語辨識：關鍵點資料的載入與前處理（訓練端）。

網站的 web/js/preprocess.js 是這裡 prep() 的 JavaScript 版，兩邊必須逐步一致，
改了其中一邊，另一邊也要跟著改。

名詞說明：
- 258 維 = MediaPipe Holistic 的姿態 33 點×4（x, y, z, 可見度）＋左手 21 點×3＋右手 21 點×3。
- 291 維 = 上面 258 維（以雙肩標準化）＋ pointing.py 算出的 33 維指向幾何。

公開版說明：原本這些函式分散在幾支實驗程式裡（tslkit、run_anchor、run_mixed、run_four），
整理公開版時把網站模型用得到的部分合併成這一支，計算步驟與參數沒有改。
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

import pointing as P

MAX_LEN = 80   # 模型輸入的幀數
FEAT = 258     # 每幀原始關鍵點數值

POSE_DIM = 33 * 4          # 0..131
LH = slice(132, 195)       # 左手 63 維
RH = slice(195, 258)       # 右手 63 維

# MediaPipe Pose 的左右對稱點配對（0 號是鼻子，沒有對稱點）。
# 索引 i 的鏡像對應點是 POSE_PAIR[i]。
POSE_PAIR = np.array(
    [0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
     18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31]
)


# --------------------------------------------------------------------------
# 左右鏡像（資料增強）
# --------------------------------------------------------------------------
def mirror(seq: np.ndarray) -> np.ndarray:
    """把一段 (T, 258) 的關鍵點序列做左右鏡像。

    正確的鏡像必須同時做三件事，少一件結果就是錯的：
      1. x 座標翻轉（x -> 1 - x）
      2. pose 的 16 組左右對稱點互換
      3. 左手 63 維區塊與右手 63 維區塊互換

    陷阱：沒偵測到的區塊是全零。若對全零套用 1 - x 會變成「全一」，
    憑空造出一個不存在的姿勢。所以只對「該幀該區塊有值」的地方翻 x。

    這個轉換是對合的：mirror(mirror(x)) == x。
    """
    seq = np.asarray(seq, dtype=np.float32)
    out = np.zeros_like(seq)

    pose = seq[:, :POSE_DIM].reshape(-1, 33, 4)
    swapped = pose[:, POSE_PAIR, :].copy()
    present = np.any(pose != 0, axis=(1, 2))
    swapped[present, :, 0] = 1.0 - swapped[present, :, 0]
    out[:, :POSE_DIM] = swapped.reshape(-1, POSE_DIM)

    for src, dst in ((LH, RH), (RH, LH)):
        hand = seq[:, src].reshape(-1, 21, 3).copy()
        present = np.any(hand != 0, axis=(1, 2))
        hand[present, :, 0] = 1.0 - hand[present, :, 0]
        out[:, dst] = hand.reshape(-1, 63)

    return out


# --------------------------------------------------------------------------
# 前處理：(T, 258) → (80, 291)
# --------------------------------------------------------------------------
def resample(seq: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    """抓出有手的區間，重取樣成固定長度。

    先找第一幀與最後一幀偵測到手的位置，只留中間那一段，
    再用最近鄰取樣成 max_len 幀（不足時重複已有的幀）。

    刻意不做線性內插：內插會在缺手的空檔兩端之間造出不存在的中間姿勢。
    最近鄰取樣的每一幀都是真的觀測到的幀。

    原專題的做法是「取前 80 幀、不足補零」，那樣比得快和比得慢的人
    在 80 格裡長得完全不一樣，長度本身就會洩漏是誰錄的。
    """
    seq = np.asarray(seq, dtype=np.float32)
    if len(seq) == 0:
        return np.zeros((max_len, FEAT), dtype=np.float32)
    has = np.any(seq[:, POSE_DIM:] != 0, axis=1)        # 有偵測到手的幀
    if not has.any():
        has = np.any(seq != 0, axis=1)                  # 退而求其次：有內容的幀
    if not has.any():
        return np.zeros((max_len, FEAT), dtype=np.float32)
    i = np.where(has)[0]
    seg = seq[i[0]:i[-1] + 1]
    take = np.round(np.linspace(0, len(seg) - 1, max_len)).astype(int)
    return seg[take].astype(np.float32)


def normalize_shoulders(seq: np.ndarray) -> np.ndarray:
    """以雙肩中點為原點、肩寬為尺度，換算每一幀的 x、y。

    只用雙肩：錄影通常只拍上半身，髖部是 MediaPipe 外推出來的，放進來等於把
    「這個人的身形」刻進座標。肩寬小於 0.05 的幀（量不出合理肩寬）原樣保留，
    沒偵測到的區塊維持全零，不能平移。
    """
    seq = np.asarray(seq, dtype=np.float32)
    out = seq.copy()
    pose = seq[:, :POSE_DIM].reshape(-1, 33, 4)
    ok = np.any(pose != 0, axis=(1, 2))
    if not np.any(ok):
        return out

    widths = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1)

    for t in range(len(seq)):
        if not ok[t]:
            continue
        p = pose[t]
        sh = p[[11, 12], :2]
        center = sh.mean(0)
        scale = widths[t]
        if scale < 0.05:                        # 量不出合理肩寬就不動它，
            continue                            # 免得除出一個放大 20 倍以上的座標
        for blk, n, dim in ((slice(0, POSE_DIM), 33, 4), (LH, 21, 3), (RH, 21, 3)):
            b = out[t, blk].reshape(n, dim)
            if not np.any(b != 0):
                continue                        # 沒偵測到的區塊維持全零，不能平移
            b[:, :2] = (b[:, :2] - center) / scale
            out[t, blk] = b.reshape(-1)
    return out


def prep(raw) -> np.ndarray:
    """原始序列（每支 (T, 258)）→ 模型輸入 (N, 80, 291)。

    順序很重要：先在原始座標上算指向幾何（要用原始的 z），再標準化前 258 維。
    網站的 preprocess.js 也是同樣的順序。
    """
    out = []
    for a in raw:
        x = resample(a)
        extra = P.add(x)[:, 258:]
        out.append(np.concatenate([normalize_shoulders(x), extra], axis=1))
    return np.stack(out)


# --------------------------------------------------------------------------
# 資料載入
# --------------------------------------------------------------------------
def signer_of(name) -> str:
    """從檔名取出示範者代碼。

    檔名格式是 {詞}_{示範者}_{批次}_{編號}.npy，例如「起床_B_0922-121120_01」→ "B"。
    所有人的片段放在同一組資料夾裡，只靠檔名區分，所以檔名一定要帶示範者代碼，
    否則沒辦法保證「考卷那個人完全沒參與訓練」。

    原專題的舊資料檔名是 clip_N（沒有示範者欄位），一律當成示範者 A；
    其他看不出示範者的檔名直接報錯，免得考卷那個人的片段被默默算進訓練。
    """
    stem = str(name).rsplit("/", 1)[-1].rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) == 4:
        return parts[1]
    if _CLIP.match(stem):          # 原專題舊資料 clip_N，一律當成 A
        return "A"
    raise ValueError(f"檔名看不出示範者：{name}（格式要是 {{詞}}_{{示範者}}_{{批次}}_{{編號}}.npy，用底線分成四段）")


_CLIP = re.compile(r"^clip_(\d+)$")


def _order(f: Path):
    """同一個資料夾裡的讀取順序：先依詞，舊格式 clip_N 依編號，其他依檔名。

    讀取順序會影響切驗證集的結果，要重現同一顆模型就不能改。
    """
    m = _CLIP.match(f.stem)
    if m:
        return (f.parent.name, 0, int(m.group(1)), "")
    return (f.parent.name, 1, 0, f.name)


def read_words(path) -> list[str]:
    """讀詞表。接受網站的 labels.json（{"0": "一起", ...}）或單純的 JSON 陣列。"""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, dict):
        return [d[str(i)] for i in range(len(d))]
    return list(d)


def load_keypoints(dirs, words=None):
    """讀關鍵點資料夾，回傳 (詞表, {示範者: (序列 list, 標籤 array)})。

    dirs 可以給好幾個資料夾，依給的順序讀。每個資料夾的結構是 {詞}/{檔名}.npy，
    每個檔是一支片段的 (幀數, 258) 陣列。words 沒給的話，用所有資料夾裡的詞資料夾名稱
    排序後當詞表；詞表以外的詞會被略過。詞表的順序就是模型輸出層的順序。
    """
    dirs = [Path(d) for d in dirs]
    if words is None:
        words = sorted({p.name for d in dirs for p in d.iterdir() if p.is_dir()})
    n2i = {w: i for i, w in enumerate(words)}
    data = {}
    for d in dirs:
        for f in sorted(d.glob("*/*.npy"), key=_order):
            w = f.parent.name
            if w not in n2i:
                continue
            a = np.load(f)
            if a.ndim != 2 or a.shape[1] != FEAT:
                continue
            raws, ys = data.setdefault(signer_of(f.name), ([], []))
            raws.append(a.astype(np.float32))
            ys.append(n2i[w])
    return list(words), {k: (v[0], np.array(v[1])) for k, v in data.items()}


def carve_val(train_idx, y, seed=0, frac=0.12):
    """從訓練資料再切一小塊當驗證集（依詞分層抽樣），只用來決定何時停止訓練。

    原專題直接把測試集當驗證集；這裡把兩者分開，驗證集只用來決定何時停止訓練。
    注意：驗證集和訓練集來自同一批錄影，會非常像，驗證正確率不能當成績。
    """
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(train_idx, test_size=frac, random_state=seed,
                              stratify=y[train_idx])
    return np.array(sorted(tr)), np.array(sorted(va))


def val_size_problem(y, frac=0.12):
    """carve_val 切不出驗證集時回傳原因（中文），切得出來就回傳 None。

    依詞分層抽樣（scikit-learn 的 train_test_split）有兩個條件：每個詞至少 2 支；
    抽出來的驗證集（總支數 × frac，無條件進位）和剩下的訓練集，支數都不能少於詞數。
    例如 25 個詞至少要 201 支。只做檢查，不影響切分結果。
    """
    y = np.asarray(y)
    n = len(y)
    if n == 0:
        return "一支片段都沒有"
    _, counts = np.unique(y, return_counts=True)
    k = len(counts)
    if counts.min() < 2:
        return f"有 {int((counts < 2).sum())} 個詞只有 1 支，每個詞至少要 2 支"
    n_val = math.ceil(frac * n)
    if n_val < k or n - n_val < k:
        need = n
        while math.ceil(frac * need) < k or need - math.ceil(frac * need) < k:
            need += 1
        return f"共 {n} 支、{k} 個詞，至少要 {need} 支（驗證集抽 {frac:.0%}，支數不能少於詞數）"
    return None


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
