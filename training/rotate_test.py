#!/usr/bin/env python3
"""換人測試：每位示範者輪流當考卷，其餘的人拿來訓練，最後列出每個人的成績與平均。

為什麼要這樣考：同一個人錄的片段彼此很像，只用「模型看過的人」測，分數會比真實使用時高很多。
輪流讓每個人都當一次「完全沒參與訓練的新使用者」，才看得出模型換人之後準不準，
以及不同人之間差多少（差很多，代表模型還很依賴某些人的打法或拍攝條件）。

用法（示範者代碼要跟檔名裡的一致，至少 2 位）：
    python training/rotate_test.py --data <關鍵點資料夾> --words web/model/labels.json \
        --signers A B C D --out output/rotate

每一輪都呼叫 train.py（同樣的前處理、模型與參數），結果存在 output/rotate/考卷X/，
最後把四輪的成績整理成 output/rotate/換人測試總表.json。
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCORE = "網站做法（鏡像平均再除以先驗）"


def main():
    ap = argparse.ArgumentParser(description="每位示範者輪流當考卷的換人測試")
    ap.add_argument("--data", nargs="+", required=True, type=Path, help="同 train.py 的 --data")
    ap.add_argument("--words", type=Path, default=None, help="同 train.py 的 --words")
    ap.add_argument("--signers", nargs="+", required=True, help="所有示範者代碼，例如 A B C D")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--out", type=Path, default=Path("output/rotate"))
    a = ap.parse_args()

    if len(a.signers) < 2:
        sys.exit("至少要 2 位示範者：一位當考卷，其他人訓練")

    rows = []
    for k in a.signers:
        others = [s for s in a.signers if s != k]
        out = a.out / f"考卷{k}"
        print(f"\n===== 考卷 {k}｜訓練 {' '.join(others)} =====", flush=True)
        cmd = [sys.executable, str(HERE / "train.py"), "--data", *map(str, a.data),
               "--train", *others, "--test", k, "--seed", str(a.seed),
               "--epochs", str(a.epochs), "--out", str(out)]
        if a.words:
            cmd += ["--words", str(a.words)]
        subprocess.run(cmd, check=True)
        rec = json.loads((out / "訓練紀錄.json").read_text(encoding="utf-8"))
        r = rec["考卷"]["結果"][SCORE]
        rows.append({"考卷": k, "訓練": others, "支數": rec["考卷"]["支數"],
                     "正確率": r["正確率"], "macroF1": r["macroF1"]})

    acc = [r["正確率"] for r in rows]
    summary = {"計分方式": SCORE, "種子": a.seed, "各輪": rows,
               "平均正確率": round(sum(acc) / len(acc), 4),
               "最低": min(acc), "最高": max(acc)}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "換人測試總表.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    print("\n換人測試結果（每位都是完全沒參與訓練的考卷）")
    for r in rows:
        print(f"  考卷 {r['考卷']:<4}{r['支數']:>6} 支　正確率 {r['正確率']:.4f}　macroF1 {r['macroF1']:.4f}")
    print(f"  平均 {summary['平均正確率']:.4f}（最低 {summary['最低']:.4f}、最高 {summary['最高']:.4f}）")
    print(f"\n總表存在 {a.out / '換人測試總表.json'}")


if __name__ == "__main__":
    main()
