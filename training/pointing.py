"""指向幾何特徵：把「指到哪裡」明確算出來，補進原本的 258 維。

為什麼需要它
------------
最近鄰居法實測（4,232 支、跨場次測試）發現錯誤高度集中在「指人」那一群：
    我 → 我們兩個 48.6%、老師 → 你 46.6%、他 → 你 36.1%
這八個詞（我/你/他/老師/去/我們兩個/你們/好）手型幾乎相同，差別只在指的方向。
八選一只有 49.2%（亂猜 12.5%）。

258 維裡有每個指節的絕對座標，理論上「方向」算得出來，
但那要模型自己從座標差去學減法與正規化。實測顯示它沒學會：
    只有雙手       全部 59.7%   指人群 49.2%
    加頭部位置     全部 59.5%                  ← 加了等於沒加
    加指向幾何     全部 64.4%   指人群 55.9%   ← +4.7 / +6.7
（這一段的數字都是最近鄰居法的結果，出自作者 2026-09-22 的實驗報告。）
所以直接把方向算好餵進去。

為什麼在增強之後才算
--------------------
鏡像增強要同時做三件事（x 翻轉、姿態左右索引互換、左右手區塊互換），
已經有 13 項測試把關（測試要用真人關鍵點，公開版未附）。如果指向特徵也進入增強流程，鏡像就得再正確處理
一整套方向向量與跨部位的相對座標——那是新的出錯機會，而且很難驗。
這裡的做法是：增強只動原本的 258 維，全部做完之後才呼叫 add()。
這樣鏡像後的序列會重新算出正確的指向，不需要額外的鏡像規則。

座標的注意事項
--------------
手部關鍵點的 z 是「相對於該手手腕」的深度，姿態的 z 是相對於髖部，
兩者不同尺度。所以：
    手內部的向量（食指根→指尖）用 x,y,z —— 同一隻手，尺度一致
    跨部位的相對位置（指尖 vs 鼻子）只用 x,y —— z 沒有共同基準
未偵測到的手，它那 15 維一律全零，和 258 維的慣例一致。
"""
import numpy as np

POSE_DIM = 132
LH = slice(132, 195)
RH = slice(195, 258)
EXTRA = 33          # 每隻手 15 + 身體 3
DIM = 258 + EXTRA

SHOULDER = (11, 12)
NOSE = 0
# 手部 21 點：0 手腕、5 食指根、8 食指尖、17 小指根、4/12/16/20 其他指尖
WRIST, IDX_MCP, IDX_TIP, PINKY_MCP = 0, 5, 8, 17
TIPS = (4, 8, 12, 16, 20)


def _unit(v, eps=1e-6):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > eps, v / np.maximum(n, eps), 0.0)


def hand_feats(hand, nose_xy, chest_xy, scale):
    """一隻手的 15 維。hand 是 (21,3)；整隻手全零就回傳全零。"""
    out = np.zeros(15, np.float32)
    if not np.any(hand):
        return out
    idx_dir = _unit(hand[IDX_TIP, :3] - hand[IDX_MCP, :3])      # 食指指向 3
    ray = _unit(hand[IDX_TIP, :3] - hand[WRIST, :3])            # 手腕→指尖 3
    palm = _unit(np.cross(hand[IDX_MCP, :3] - hand[WRIST, :3],  # 手掌法向量 3
                          hand[PINKY_MCP, :3] - hand[WRIST, :3]))
    tip = hand[IDX_TIP, :2]
    rel_nose = (tip - nose_xy) / scale                          # 指尖 vs 鼻子 2
    rel_chest = (tip - chest_xy) / scale                        # 指尖 vs 胸口 2
    # 指向自己的程度：食指的平面方向 與 指尖→胸口方向 的內積，+1 是直指自己。
    # 特例：指尖幾乎貼在胸口時（「我」就是手指點自己胸口），
    # 指尖→胸口這個向量長度趨近 0，正規化後變成 0，內積也變 0——
    # 最需要它作用的那個詞剛好會失效。所以近距離直接判定為 +1：
    # 指尖都碰到胸口了，那本來就是指自己，不需要再看方向。
    d_chest = float(np.linalg.norm(chest_xy - tip) / scale)
    if d_chest < 0.25:
        inward = 1.0
    else:
        inward = float(np.dot(_unit(idx_dir[:2]), _unit(chest_xy - tip)))
    # 手張開程度：五個指尖到手腕的平均距離 ÷ 手腕到中指根的距離。
    # 握拳接近 1，張開明顯大於 1。手型的粗略代理。
    palm_len = np.linalg.norm(hand[9, :3] - hand[WRIST, :3])
    spread = (float(np.mean(np.linalg.norm(hand[list(TIPS), :3] - hand[WRIST, :3], axis=1))
                    / max(palm_len, 1e-6)) if palm_len > 1e-6 else 0.0)
    out[:3] = idx_dir
    out[3:6] = ray
    out[6:9] = palm
    out[9:11] = rel_nose
    out[11:13] = rel_chest
    out[13] = inward
    out[14] = spread
    return out


def add(seq):
    """輸入 (T,258)，輸出 (T,291)。原本的 258 維原封不動接在前面。"""
    seq = np.asarray(seq, dtype=np.float32)
    T = len(seq)
    extra = np.zeros((T, EXTRA), np.float32)
    pose = seq[:, :POSE_DIM].reshape(T, 33, 4)

    for t in range(T):
        p = pose[t]
        if not np.any(p):
            continue                                   # 沒有姿態就沒有身體參考框架
        sh = p[list(SHOULDER), :2]
        chest = sh.mean(0)
        scale = float(np.linalg.norm(sh[0] - sh[1]))
        if scale < 0.05:
            continue                                   # 肩寬量不出來，除下去會爆掉
        nose = p[NOSE, :2]
        lh = seq[t, LH].reshape(21, 3)
        rh = seq[t, RH].reshape(21, 3)
        extra[t, 0:15] = hand_feats(lh, nose, chest, scale)
        extra[t, 15:30] = hand_feats(rh, nose, chest, scale)
        extra[t, 30:32] = _unit(sh[0] - sh[1])         # 肩膀連線方向：身體有沒有轉
        if np.any(lh) and np.any(rh):                  # 兩手指尖距離
            extra[t, 32] = np.linalg.norm(lh[IDX_TIP, :2] - rh[IDX_TIP, :2]) / scale
    return np.concatenate([seq, extra], axis=1)


def add_batch(X):
    """(N,T,258) → (N,T,291)"""
    return np.stack([add(x) for x in X])
