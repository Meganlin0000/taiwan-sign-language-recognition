/**
 * 前處理：把 MediaPipe 逐幀的 258 維，變成新模型要的 (80, 291)。
 *
 * 這四步是逐行照 Python 搬過來的，順序不能動：
 *
 *   1. resample    tslkit.resample             抓「有手」的區間，最近鄰重取樣成 80 幀
 *   2. pointing    pointing.add                在<原始座標>上算 33 維指向幾何
 *   3. normalize   tslkit.normalize_shoulders  只有肩：前 258 維以雙肩中點為原點、肩寬為尺度
 *   4. concat      tslkit.prep                 每一列 = [normalize 的 258 維, pointing 的 33 維]
 *
 * 第 2 步一定要在第 3 步之前：指向幾何要用原始座標的 z，而且 pointing.add
 * 自己會用肩寬做尺度。這跟 training/tslkit.py 的 prep() 順序一樣：
 *     x = resample(a); extra = P.add(x)[:, 258:]; concat(normalize_shoulders(x), extra)
 *
 * 搬錯一個索引不會報錯，只會靜默失準。所以這個檔案有一支對帳測試
 * （公開版未附，因為要用到真人關鍵點）：同一批真實片段，Python 與這裡各算一次，
 * 逐數字比對；差異超過 1e-4 就不准部署。
 */

export const RAW_DIM = 258;        // MediaPipe 逐幀：姿態 33×4 + 左手 21×3 + 右手 21×3
export const EXTRA_DIM = 33;       // 指向幾何：每隻手 15 + 身體 3
export const MODEL_DIM = RAW_DIM + EXTRA_DIM;   // 291
export const ROWS = 80;

const POSE_DIM = 132;
const LH = 132, RH = 195;
const NOSE = 0, L_SHOULDER = 11, R_SHOULDER = 12;
const WRIST = 0, IDX_MCP = 5, IDX_TIP = 8, MID_MCP = 9, PINKY_MCP = 17;
const TIPS = [4, 8, 12, 16, 20];
const EPS = 1e-6;

/* ── 0. 鏡頭長寬比校正 ──────────────────────────────────────
 *
 * MediaPipe 的 x 是「除以畫面寬」、y 是「除以畫面高」。而這個檔案的尺度
 * 只有一個：肩寬，它量出來是 x 方向的長度。於是「垂直距離 ÷ 肩寬」會多乘
 * 一個「畫面寬 ÷ 畫面高」：16:9 是 1.78、4:3 是 1.33、手機直拿 3:4 是 0.75。
 * 同一個人比同一個詞，換一台鏡頭，送進模型的垂直座標就整體縮放——
 * 模型沒見過那個尺度，於是 4:3 明顯比 16:9 不準、3:4 認錯暴增。
 *
 * 修法：把 y 乘上 REF_ASPECT ÷（實際畫面寬 ÷ 高），等於把座標換算成
 * 「如果這台鏡頭是 REF_ASPECT」的樣子。只動 y：
 *   - x 本來就是寬的比例，與長寬比無關。
 *   - z 的尺度跟 x 相同（也是除以畫面寬），與長寬比無關，所以不能動。
 * 下游全部是差值（normalizeShoulders 減肩中點、pointing 算相對向量），
 * 所以直接整體乘 k 與「以畫面中心縮放」等價。
 *
 * 只作用在「送進模型的那一份」：呼叫點是 buildModelInput，
 * continuous-core.js 的窗格、動作量門檻（minTipTravel）看到的還是原始座標，
 * 門檻值不會被這個縮放連帶改變。
 */
export const REF_ASPECT = 16 / 9;      // 換算基準。B、C、D 與 A 補錄的 300 支都是 16:9（1280×720）；A 舊的 4,232 支是 4:3（640×480）。
let aspectScaleY = 1;

/** 畫面尺寸變了就呼叫一次（app.js 的 syncDimensions）。回傳實際採用的 k。 */
export function setFrameAspect(width, height, ref = REF_ASPECT) {
  aspectScaleY = (width > 0 && height > 0 && ref > 0) ? ref / (width / height) : 1;
  return aspectScaleY;
}
export function frameAspectScaleY() { return aspectScaleY; }

/** 就地把 80×258 裡所有的 y 乘上 k。全零區塊乘完還是全零，不會憑空長出座標。 */
function applyAspectY(raw, rows = ROWS) {
  const k = aspectScaleY;
  if (!(k > 0) || Math.abs(k - 1) < 1e-9) return raw;
  for (let t = 0; t < rows; t += 1) {
    const base = t * RAW_DIM;
    for (let j = 0; j < 33; j += 1) raw[base + j * 4 + 1] *= k;          // 姿態 y（z 與可見度不動）
    for (const off of [LH, RH]) {
      for (let j = 0; j < 21; j += 1) raw[base + off + j * 3 + 1] *= k;  // 手部 y
    }
  }
  return raw;
}

/* ── 小工具 ─────────────────────────────────────────────── */

function anyNonZero(f, from, to) {
  for (let i = from; i < to; i += 1) if (f[i] !== 0) return true;
  return false;
}

/** numpy 的 np.round：四捨六入、五取偶數。JS 的 Math.round 是五進位，兩者在 .5 時不同。 */
function roundHalfEven(x) {
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return (f % 2 === 0) ? f : f + 1;
}

/** np.linspace(0, n-1, rows) 的第 r 個值。照 numpy 的算法：step = (n-1)/(rows-1)，y = r*step，最後一個直接設成 n-1。 */
function linspaceAt(n, rows, r) {
  if (rows <= 1) return 0;
  if (r === rows - 1) return n - 1;
  const step = (n - 1) / (rows - 1);
  return r * step;
}

function norm3(ax, ay, az) { return Math.sqrt(ax * ax + ay * ay + az * az); }
function norm2(ax, ay) { return Math.sqrt(ax * ax + ay * ay); }

/** pointing._unit：長度 > eps 才正規化，否則回零向量。 */
function unit3(v) {
  const n = norm3(v[0], v[1], v[2]);
  if (!(n > EPS)) return [0, 0, 0];
  const d = Math.max(n, EPS);
  return [v[0] / d, v[1] / d, v[2] / d];
}
function unit2(v) {
  const n = norm2(v[0], v[1]);
  if (!(n > EPS)) return [0, 0];
  const d = Math.max(n, EPS);
  return [v[0] / d, v[1] / d];
}

/* ── 1. 重取樣 ──────────────────────────────────────────── */

/**
 * tslkit.resample 的 JS 版。
 * frames 是一串 Float32Array(258)，每一幀都要在（包含手沒被偵測到的幀——
 * 那些幀手部欄位是零、姿態還在，訓練資料就是這樣）。
 * 回傳 Float32Array(80*258)，列優先。
 */
export function resampleActive(frames, rows = ROWS) {
  const out = new Float32Array(rows * RAW_DIM);
  if (!frames.length) return out;
  // 有偵測到手的幀（手部欄位 132..257 有非零）
  let has = frames.map(f => anyNonZero(f, POSE_DIM, RAW_DIM));
  if (!has.some(Boolean)) has = frames.map(f => anyNonZero(f, 0, RAW_DIM));
  if (!has.some(Boolean)) return out;
  const first = has.indexOf(true), last = has.lastIndexOf(true);
  const seg = frames.slice(first, last + 1);
  const n = seg.length;
  for (let r = 0; r < rows; r += 1) {
    const j = roundHalfEven(linspaceAt(n, rows, r));
    out.set(seg[Math.min(j, n - 1)], r * RAW_DIM);
  }
  return out;
}

/* ── 2. 指向幾何 ────────────────────────────────────────── */

/** pointing.hand_feats：一隻手的 15 維。hand 是這一幀的 63 個數（21×3）。 */
function handFeats(seq, base, noseX, noseY, chestX, chestY, scale, out, o) {
  // 整隻手全零就維持全零
  if (!anyNonZero(seq, base, base + 63)) return;
  const P = k => [seq[base + k * 3], seq[base + k * 3 + 1], seq[base + k * 3 + 2]];
  const tip = P(IDX_TIP), mcp = P(IDX_MCP), wrist = P(WRIST), pinky = P(PINKY_MCP);

  const idxDir = unit3([tip[0] - mcp[0], tip[1] - mcp[1], tip[2] - mcp[2]]);
  const ray = unit3([tip[0] - wrist[0], tip[1] - wrist[1], tip[2] - wrist[2]]);
  const a = [mcp[0] - wrist[0], mcp[1] - wrist[1], mcp[2] - wrist[2]];
  const b = [pinky[0] - wrist[0], pinky[1] - wrist[1], pinky[2] - wrist[2]];
  const palm = unit3([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]);

  const relNoseX = (tip[0] - noseX) / scale, relNoseY = (tip[1] - noseY) / scale;
  const relChestX = (tip[0] - chestX) / scale, relChestY = (tip[1] - chestY) / scale;

  // 指向自己的程度。指尖幾乎貼在胸口（「我」）時向量長度趨近 0，
  // 正規化後會變 0——最需要它的詞剛好失效，所以近距離直接判 +1。
  const dChest = norm2(chestX - tip[0], chestY - tip[1]) / scale;
  let inward;
  if (dChest < 0.25) inward = 1.0;
  else {
    const u = unit2([idxDir[0], idxDir[1]]);
    const v = unit2([chestX - tip[0], chestY - tip[1]]);
    inward = u[0] * v[0] + u[1] * v[1];
  }

  // 手張開程度：五指尖到手腕的平均距離 ÷ 手腕到中指根的距離
  const mid = P(MID_MCP);
  const palmLen = norm3(mid[0] - wrist[0], mid[1] - wrist[1], mid[2] - wrist[2]);
  let spread = 0;
  if (palmLen > EPS) {
    let s = 0;
    for (const k of TIPS) { const t = P(k); s += norm3(t[0] - wrist[0], t[1] - wrist[1], t[2] - wrist[2]); }
    spread = (s / TIPS.length) / Math.max(palmLen, EPS);
  }

  out[o] = idxDir[0]; out[o + 1] = idxDir[1]; out[o + 2] = idxDir[2];
  out[o + 3] = ray[0]; out[o + 4] = ray[1]; out[o + 5] = ray[2];
  out[o + 6] = palm[0]; out[o + 7] = palm[1]; out[o + 8] = palm[2];
  out[o + 9] = relNoseX; out[o + 10] = relNoseY;
  out[o + 11] = relChestX; out[o + 12] = relChestY;
  out[o + 13] = inward;
  out[o + 14] = spread;
}

/**
 * pointing.add 的 JS 版：輸入 80×258（原始座標，<尚未>標準化），輸出 80×33。
 */
export function pointingFeatures(raw, rows = ROWS) {
  const extra = new Float32Array(rows * EXTRA_DIM);
  for (let t = 0; t < rows; t += 1) {
    const base = t * RAW_DIM;
    if (!anyNonZero(raw, base, base + POSE_DIM)) continue;     // 沒有姿態就沒有身體參考框架
    const sx0 = raw[base + L_SHOULDER * 4], sy0 = raw[base + L_SHOULDER * 4 + 1];
    const sx1 = raw[base + R_SHOULDER * 4], sy1 = raw[base + R_SHOULDER * 4 + 1];
    const chestX = (sx0 + sx1) / 2, chestY = (sy0 + sy1) / 2;
    const scale = norm2(sx0 - sx1, sy0 - sy1);
    if (scale < 0.05) continue;                                 // 肩寬量不出來，除下去會爆
    const noseX = raw[base + NOSE * 4], noseY = raw[base + NOSE * 4 + 1];
    const o = t * EXTRA_DIM;
    handFeats(raw, base + LH, noseX, noseY, chestX, chestY, scale, extra, o);
    handFeats(raw, base + RH, noseX, noseY, chestX, chestY, scale, extra, o + 15);
    const dir = unit2([sx0 - sx1, sy0 - sy1]);                  // 肩膀連線方向：身體有沒有轉
    extra[o + 30] = dir[0]; extra[o + 31] = dir[1];
    const hasL = anyNonZero(raw, base + LH, base + LH + 63);
    const hasR = anyNonZero(raw, base + RH, base + RH + 63);
    if (hasL && hasR) {                                         // 兩手指尖距離
      const lx = raw[base + LH + IDX_TIP * 3], ly = raw[base + LH + IDX_TIP * 3 + 1];
      const rx = raw[base + RH + IDX_TIP * 3], ry = raw[base + RH + IDX_TIP * 3 + 1];
      extra[o + 32] = norm2(lx - rx, ly - ry) / scale;
    }
  }
  return extra;
}

/* ── 3. 雙肩標準化 ──────────────────────────────────────── */

/**
 * tslkit.normalize_shoulders 的 JS 版。回傳新的 80×258（不改輸入）。
 * 每一幀：中心 = 雙肩中點、尺度 = 該幀肩寬；只動 x,y，z 與可見度不動；
 * 沒偵測到的區塊維持全零（不能平移零）。
 */
export function normalizeShoulders(raw, rows = ROWS) {
  const out = raw.slice();
  for (let t = 0; t < rows; t += 1) {
    const base = t * RAW_DIM;
    if (!anyNonZero(raw, base, base + POSE_DIM)) continue;
    const sx0 = raw[base + L_SHOULDER * 4], sy0 = raw[base + L_SHOULDER * 4 + 1];
    const sx1 = raw[base + R_SHOULDER * 4], sy1 = raw[base + R_SHOULDER * 4 + 1];
    const scale = norm2(sx0 - sx1, sy0 - sy1);
    if (scale < 0.05) continue;
    const cx = (sx0 + sx1) / 2, cy = (sy0 + sy1) / 2;
    // 姿態 33 點 × 4
    for (let k = 0; k < 33; k += 1) {
      const i = base + k * 4;
      out[i] = (raw[i] - cx) / scale;
      out[i + 1] = (raw[i + 1] - cy) / scale;
    }
    for (const off of [LH, RH]) {
      if (!anyNonZero(raw, base + off, base + off + 63)) continue;
      for (let k = 0; k < 21; k += 1) {
        const i = base + off + k * 3;
        out[i] = (raw[i] - cx) / scale;
        out[i + 1] = (raw[i + 1] - cy) / scale;
      }
    }
  }
  return out;
}

/* ── 4. 組合 ───────────────────────────────────────────── */

/**
 * frames（一串 Float32Array(258)，含沒偵測到手的幀）→ Float32Array(80*291)。
 * 這就是 tslkit.prep 對一支片段做的事。
 */
export function buildModelInput(frames) {
  // resampleActive 回傳的是新陣列，applyAspectY 就地改它不會動到窗格裡存的影格。
  const raw = applyAspectY(resampleActive(frames));
  const extra = pointingFeatures(raw);
  const norm = normalizeShoulders(raw);
  const out = new Float32Array(ROWS * MODEL_DIM);
  for (let r = 0; r < ROWS; r += 1) {
    out.set(norm.subarray(r * RAW_DIM, (r + 1) * RAW_DIM), r * MODEL_DIM);
    out.set(extra.subarray(r * EXTRA_DIM, (r + 1) * EXTRA_DIM), r * MODEL_DIM + RAW_DIM);
  }
  return out;
}

/* ── 左右鏡像（測試時增強用）───────────────────────────── */

/** MediaPipe Pose 的左右對稱點配對；索引 i 的鏡像對應點是 POSE_PAIR[i]。0 是鼻子，沒有對稱點。 */
const POSE_PAIR = [0, 4, 5, 6, 1, 2, 3, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15,
                   18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31];

/**
 * 把一幀 258 維做左右鏡像，逐行照 tslkit.mirror 搬。
 * 三件事要同時做，少一件結果就是錯的：
 *   1. x 座標翻轉（x → 1 − x）
 *   2. 姿態的 16 組左右對稱點互換
 *   3. 左手與右手的 63 維區塊互換
 * 陷阱：沒偵測到的區塊是全零，對全零做 1 − x 會變成「全一」，
 * 憑空造出一個不存在的姿勢。所以只對「該區塊有值」的地方翻 x。
 */
export function mirrorFrame(f) {
  const out = new Float32Array(RAW_DIM);
  const posePresent = anyNonZero(f, 0, POSE_DIM);
  for (let k = 0; k < 33; k += 1) {
    const src = POSE_PAIR[k] * 4, dst = k * 4;
    out[dst] = posePresent ? 1 - f[src] : f[src];
    out[dst + 1] = f[src + 1];
    out[dst + 2] = f[src + 2];
    out[dst + 3] = f[src + 3];
  }
  for (const [src, dst] of [[LH, RH], [RH, LH]]) {
    if (!anyNonZero(f, src, src + 63)) continue;      // 全零區塊維持全零
    for (let k = 0; k < 21; k += 1) {
      out[dst + k * 3] = 1 - f[src + k * 3];
      out[dst + k * 3 + 1] = f[src + k * 3 + 1];
      out[dst + k * 3 + 2] = f[src + k * 3 + 2];
    }
  }
  return out;
}

/** 同一串影格的鏡像版模型輸入。測試時增強：正反各推論一次再平均。 */
export function buildMirroredInput(frames) {
  return buildModelInput(frames.map(mirrorFrame));
}
