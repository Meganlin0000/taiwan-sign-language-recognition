#!/usr/bin/env python3
"""Keras 模型（.keras）→ TensorFlow.js Layers 格式，並修正 Keras 3 與 TFJS 4.22 的序列化差異。

用法（要在轉檔環境裡跑，見 requirements-convert.txt）：
    python tools/轉檔_keras轉tfjs.py <模型.keras> <輸出目錄>
    ⚠ 輸出目錄會先整個清空，不要直接指到 web/model/；轉好再把 model.json 和 .bin 複製過去。
    python tools/轉檔_keras轉tfjs.py --normalize-only <已轉檔的目錄>     只做格式修正，不需要 tensorflow

為什麼需要「修正」這一步：官方轉檔器 tensorflowjs 4.22 是對著 Keras 2 寫的，
而訓練環境是 Keras 3。轉出來的 model.json 有三處 TFJS 讀不懂：
  一、InputLayer 的形狀鍵叫 batch_shape，TFJS 只認 batch_input_shape
      （不修會報 An InputLayer should be passed either a batchInputShape or an inputShape）
  二、dtype 是一個 DTypePolicy 物件，TFJS 要一個字串 "float32"
  三、LSTM 的權重路徑多一層 cell（lstm/lstm_cell/kernel），TFJS 要 lstm/kernel
      不修的話載入時會找不到權重

轉檔器還會硬拉 tensorflow_decision_forests，那條相依鏈跟 protobuf 版本衝突，
但 Keras→TFJS 的路徑根本用不到它。這裡塞一個空模組讓匯入過。
"""
import json
import os
import re
import shutil
import sys
import types
import warnings


def normalize_topology(model_json_path):
    m = json.load(open(model_json_path, encoding="utf-8"))
    changes = []
    topo = m["modelTopology"]
    model_cfg = topo.get("model_config", topo)
    layers = model_cfg["config"]["layers"]
    for layer in layers:
        cfg = layer["config"]
        if "batch_shape" in cfg and "batch_input_shape" not in cfg:
            cfg["batch_input_shape"] = cfg.pop("batch_shape")
            changes.append(f"{layer['class_name']}: batch_shape → batch_input_shape")
        d = cfg.get("dtype")
        if isinstance(d, dict):
            name = (d.get("config") or {}).get("name") or "float32"
            cfg["dtype"] = name
            changes.append(f"{layer['class_name']}: dtype policy → \"{name}\"")
    for group in m["weightsManifest"]:
        for w in group["weights"]:
            new = re.sub(r"/[^/]*cell[^/]*/", "/", w["name"])
            if new != w["name"]:
                changes.append(f"權重 {w['name']} → {new}")
                w["name"] = new
    json.dump(m, open(model_json_path, "w", encoding="utf-8"))
    return changes, layers, [w["name"] for g in m["weightsManifest"] for w in g["weights"]]


def convert(keras_path, out_dir):
    warnings.filterwarnings("ignore")
    sys.modules["tensorflow_decision_forests"] = types.ModuleType("tensorflow_decision_forests")
    from tensorflowjs.converters import save_keras_model   # noqa: E402
    import tensorflow as tf                                 # noqa: E402
    model = tf.keras.models.load_model(keras_path, compile=False)
    print(f"模型 {os.path.basename(keras_path)}　輸入 {model.input_shape}　輸出 {model.output_shape}　參數 {model.count_params():,}")
    shutil.rmtree(out_dir, ignore_errors=True)
    save_keras_model(model, out_dir)
    return normalize_topology(os.path.join(out_dir, "model.json"))


def report(changes, layers, names, out_dir):
    print(f"修正 {len(changes)} 處：")
    for c in changes[:12]:
        print("  ", c)
    if len(changes) > 12:
        print(f"   …另 {len(changes) - 12} 處")
    first = layers[0]["config"]
    print(f"第一層 {layers[0]['class_name']}　batch_input_shape={first.get('batch_input_shape')}　dtype={first.get('dtype')}")
    print(f"權重 {len(names)} 個，例如 {names[:3]}")
    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}  {os.path.getsize(os.path.join(out_dir, f)):,} bytes")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--normalize-only":
        out = sys.argv[2]
        report(*normalize_topology(os.path.join(out, "model.json")), out)
    elif len(sys.argv) == 3:
        report(*convert(sys.argv[1], sys.argv[2]), sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
