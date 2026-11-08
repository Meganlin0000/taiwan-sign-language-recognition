#!/usr/bin/env python3
"""訓練網站用的手語辨識模型（三層 LSTM），存成 .keras，並附上網站要的 labels.json 與 class-prior.json。

網站目前那顆模型就是用這支程式的設定訓練的：
    python training/train.py --data <A 的舊關鍵點> <新關鍵點> --words web/model/labels.json \
        --train A B C --test D --seed 42 --out output/model

流程
  1. 讀關鍵點（每支片段一個 (幀數, 258) 的 .npy，檔名帶示範者代碼，見 tslkit.load_keypoints）
  2. 每位訓練示範者各自依詞分層抽 12% 當驗證集（只用來決定何時停止，不是成績）
  3. 訓練資料加上左右鏡像的副本
  4. 前處理成 (80, 291)，訓練三層 LSTM，驗證正確率 20 輪沒進步就停，留最好的那一輪
  5. 有指定 --test 時，用那位「完全沒參與訓練」的示範者當考卷評分

考卷一定要是沒參與訓練的人。同一場錄影切出來的片段彼此很像，
隨機切分會讓它們同時出現在訓練與測試裡，分數會嚴重高估。

公開版說明：原本是 finalize_four.py 加上它引用的幾支實驗程式，整理公開版時合併成這一支；
資料排列順序、隨機種子、模型架構與訓練參數都沒有改。另外加上每一輪的紀錄檔（CSV），
那只是記錄，不影響訓練結果。
"""
import argparse
import csv
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402

from tslkit import (carve_val, load_keypoints, mirror, prep, read_words,  # noqa: E402
                    save_json, val_size_problem)


def build(n_classes, input_shape, seed=42):
    """模型架構（沿用 2025 原專題的架構，網站的 TensorFlow.js 版讀的就是這個結構）。"""
    tf.keras.utils.set_random_seed(seed)
    m = tf.keras.Sequential([
        tf.keras.layers.Masking(mask_value=0.0, input_shape=input_shape),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(n_classes, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.AdamW(1e-3),
              loss="categorical_crossentropy", metrics=["accuracy"])
    return m


def build_training(data, signers, seed):
    """組訓練集與驗證集。順序是 A、A 鏡像、B、B 鏡像……，改了順序就無法重現同一顆模型。"""
    tr_raw, va_X, va_y = [], [], []
    for k in signers:
        raws, y = data[k]
        t2, va = carve_val(np.arange(len(y)), y, seed=seed)
        tr_raw.append(([raws[i] for i in t2], y[t2]))
        va_X.append(prep([raws[i] for i in va]))
        va_y.append(y[va])
    Xs, ys = [], []
    for raws, y in tr_raw:
        Xs.append(prep(raws)); ys.append(y)                       # 原始
        Xs.append(prep([mirror(r) for r in raws])); ys.append(y)  # 左右鏡像
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(va_X), np.concatenate(va_y))


class EpochLog(tf.keras.callbacks.Callback):
    """每一輪印一行，並記下花幾秒。只做記錄，不影響訓練。"""

    def on_train_begin(self, logs=None):
        self.seconds = []

    def on_epoch_begin(self, epoch, logs=None):
        self._t = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.seconds.append(round(time.time() - self._t, 2))
        logs = logs or {}
        print(f"  第 {epoch + 1:>3} 輪　訓練 {logs.get('accuracy', 0):.4f}　"
              f"驗證 {logs.get('val_accuracy', 0):.4f}　{self.seconds[-1]:.0f} 秒", flush=True)


def macro_f1(y_true, y_pred, n):
    f1 = []
    for c in range(n):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        if tp + fn == 0:
            continue                       # 這個詞沒有出現在考卷裡，不列入平均
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn)
        f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return float(np.mean(f1)) if f1 else 0.0


def write_confusion(path, y_true, y_pred, words):
    n = len(words)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["正確答案＼被判成"] + list(words))
        for i in range(n):
            w.writerow([words[i]] + [int(v) for v in cm[i]])


def main():
    ap = argparse.ArgumentParser(description="訓練手語辨識模型")
    ap.add_argument("--data", nargs="+", required=True, type=Path,
                    help="關鍵點資料夾，可以給多個；結構是 {詞}/{詞}_{示範者}_{批次}_{編號}.npy")
    ap.add_argument("--words", type=Path, default=None,
                    help="詞表（例如 web/model/labels.json）；沒給就用資料夾裡的詞名排序")
    ap.add_argument("--train", nargs="+", required=True, help="訓練用的示範者代碼，例如 A B C")
    ap.add_argument("--test", default=None, help="當考卷的示範者代碼（完全不參與訓練），例如 D")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("output/model"))
    a = ap.parse_args()

    if a.test and a.test in a.train:
        sys.exit("考卷示範者不能同時出現在訓練示範者裡")
    words, data = load_keypoints(a.data, read_words(a.words) if a.words else None)
    n = len(words)
    print(f"詞 {n} 個")
    for k in sorted(data):
        mark = "訓練" if k in a.train else ("考卷" if k == a.test else "不使用")
        print(f"  示範者 {k:<6}{len(data[k][0]):>6} 支　{mark}")
    missing = [k for k in a.train + ([a.test] if a.test else []) if k not in data]
    if missing:
        sys.exit(f"找不到這些示範者的資料：{missing}")
    for k in a.train:                      # 先檢查，免得前處理跑完才在切驗證集時報英文錯誤
        why = val_size_problem(data[k][1])
        if why:
            sys.exit(f"示範者 {k} 的片段不夠切出驗證集：{why}")
    # 詞表裡有詞在訓練示範者那邊一支都沒有時，先驗是 0，網站算法的分數會變成 NaN
    counts = np.bincount(np.concatenate([data[k][1] for k in a.train]), minlength=n)
    if (counts == 0).any():
        sys.exit("這些詞在訓練示範者的片段裡一支都沒有，請補錄，或把它們移出 --words 詞表："
                 + "、".join(w for w, c in zip(words, counts) if c == 0))

    print("\n前處理中…")
    t0 = time.time()
    Xtr, ytr, Xva, yva = build_training(data, a.train, a.seed)
    print(f"  完成（{time.time() - t0:.0f} 秒）　訓練 {len(ytr)}（含鏡像）／驗證 {len(yva)}　"
          f"輸入 {Xtr.shape[1]}×{Xtr.shape[2]}")

    a.out.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(a.seed)
    model = build(n, Xtr.shape[1:], a.seed)
    log = EpochLog()
    t1 = time.time()
    hist = model.fit(Xtr, tf.keras.utils.to_categorical(ytr, n),
                     validation_data=(Xva, tf.keras.utils.to_categorical(yva, n)),
                     epochs=a.epochs, batch_size=32, verbose=0,
                     callbacks=[tf.keras.callbacks.EarlyStopping(
                                    monitor="val_accuracy", patience=20,
                                    restore_best_weights=True, mode="max"),
                                tf.keras.callbacks.CSVLogger(str(a.out / "每一輪.csv")),
                                log])
    h = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    best = int(np.argmax(h["val_accuracy"])) + 1
    print(f"訓練完成：練了 {len(h['loss'])} 輪，最好的是第 {best} 輪"
          f"（驗證 {h['val_accuracy'][best - 1]:.4f}），{time.time() - t1:.0f} 秒")

    model.save(a.out / "model.keras")
    # 網站要的兩個檔：詞序（輸出層索引 → 詞）與類別先驗（訓練示範者全部片段裡各詞的比例，
    # 網站把模型輸出除以它再正規化，避免輸入一模糊就往樣本多的詞倒）。
    save_json({str(i): w for i, w in enumerate(words)}, a.out / "labels.json")
    (a.out / "class-prior.json").write_text(json.dumps({
        "說明": "訓練集的類別比例。網站把模型輸出除以先驗再正規化（α=1）。",
        "訓練示範者": a.train, "訓練支數": int(counts.sum()), "alpha": 1.0,
        "詞序": words, "先驗": [round(float(c / counts.sum()), 6) for c in counts],
    }, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    record = {"詞序": words, "訓練示範者": a.train, "考卷示範者": a.test, "種子": a.seed,
              "訓練筆數（含鏡像）": int(len(ytr)), "驗證筆數": int(len(yva)),
              "練了幾輪": len(h["loss"]), "最好的一輪": best,
              "最好那輪的驗證正確率": round(h["val_accuracy"][best - 1], 4),
              "總秒數": round(time.time() - t1, 1), "每一輪秒數": log.seconds,
              "環境": {"Python": platform.python_version(), "TensorFlow": tf.__version__,
                       "Keras": __import__("keras").__version__, "NumPy": np.__version__,
                       "電腦": platform.platform()}}

    if a.test:
        raws, yte = data[a.test]
        p = model.predict(prep(raws), batch_size=256, verbose=0)
        pm = model.predict(prep([mirror(r) for r in raws]), batch_size=256, verbose=0)
        prior = counts / counts.sum()
        site = (p + pm) / 2 / prior
        site = site / site.sum(1, keepdims=True)
        res = {}
        for name, prob in (("單純", p), ("原圖與鏡像平均", (p + pm) / 2),
                           ("網站做法（鏡像平均再除以先驗）", site)):
            pred = prob.argmax(1)
            res[name] = {"正確率": round(float((pred == yte).mean()), 4),
                         "macroF1": round(macro_f1(yte, pred, n), 4)}
        write_confusion(a.out / f"考卷{a.test}_混淆矩陣.csv", yte, site.argmax(1), words)
        record["考卷"] = {"支數": int(len(yte)), "結果": res}
        print(f"\n考卷 {a.test}（{len(yte)} 支，完全沒參與訓練）：")
        for k, v in res.items():
            print(f"  {k:<18}正確率 {v['正確率']:.4f}　macroF1 {v['macroF1']:.4f}")

    save_json(record, a.out / "訓練紀錄.json")
    saved = "model.keras、labels.json、class-prior.json、每一輪.csv、訓練紀錄.json"
    if a.test:
        saved += f"、考卷{a.test}_混淆矩陣.csv"
    print(f"\n已存到 {a.out}/：{saved}")


if __name__ == "__main__":
    main()
