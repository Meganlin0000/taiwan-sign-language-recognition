/**
 * 連續辨識的滑動窗格，以及「結果穩定才記下」的判斷，給 25 個詞的分類器用。
 * 純 JavaScript：不碰鏡頭、網路、儲存，不看使用者選了哪個示範詞，也不改模型。
 * 所有時間都要用同一個單調遞增的毫秒時鐘（例如 performance.now()）。
 *
 * 2026-09-23 稽核後的四項修正，每一項都對應一個實際會發生的錯誤：
 *
 *   一、動作量門檻（minTipTravel）
 *       原本只要「偵測得到手 + 連續三次同一個詞 + 信心 ≥ 門檻」就記錄，
 *       沒有任何一項檢查「手到底有沒有在動」。而模型只有 25 類、沒有
 *       「沒在比」這一類，所以一隻靜止的手一定會被分到某個詞去；
 *       實測靜止的手常常拿到 ≥ 0.90 的最高信心。結果是使用者比完一個詞
 *       停下來想下一句，句子就自己多出字。
 *
 *   二、確認後清窗格並進入冷卻（confirmCooldownMs）
 *       原本確認之後窗格原封不動，前一個詞的影格還留在裡面跟下一個詞
 *       的開頭混在一起，兩個詞的交界會被判成第三個詞。
 *
 *   三、同詞改用冷卻，不再永久封鎖
 *       原本用 _lastConfirmedIndex 擋「同一個詞連續記錄」，但那個記憶
 *       只有手離開畫面滿 700 毫秒才清掉。於是「我再比一次給您看」
 *       第二次永遠不會被記錄，畫面卻還是顯示「已辨識」。
 *
 *   四、依實測影格率調整門檻
 *       原本 minFrames 固定 12、窗格 2500 毫秒。裝置慢到每秒不足約 4.8 幀時，
 *       窗格裡根本放不進 12 張有效影格，條件在數學上永遠不成立，
 *       畫面會無限停在「正在觀察動作」而且不給任何理由。
 *       2026-09-25 補：容量換算還不夠。每秒 8 格時容量算出來剛好是 12，門檻沒有放寬，
 *       但 12 張在那個速度下等於 1,500 毫秒的手部畫面，比一個手語動作還久，
 *       所以短詞連「放下手補判」都拿不到輸入。改用 minFramesMs（見下）。
 */

import { buildModelInput, buildMirroredInput, MODEL_DIM, RAW_DIM, frameAspectScaleY } from './preprocess.js';

/** 食指尖在手部 21 點裡的編號。 */
const INDEX_TIP = 8;
/** 兩隻手在 258 維特徵裡的起始欄位。 */
const HAND_OFFSETS = [132, 195];

/**
 * 2026-09-29「同學」手形關卡。
 * 模型把「一起」與「同學」看成互為前半段：比「一起」到一半，模型就以 0.92–1.00 認定是「同學」，
 * 連續三次一湊滿就被記下（9/25 實測 17 次一起→同學裡 12 次是這樣）。
 * 「同學」有「一起」沒有的手形：大拇指與小指伸直、其餘三指彎曲。
 * 用四位示範者的關鍵點量過（y 已換算成 16:9 參考比例，與網站前處理相同）：
 *   同學片段看得到這個手形（≥2 幀）：示範者 B 76/79、C 15/22、D 15/15、A 98/156；
 *   一起片段偶爾也抓得到 16/235，但關卡只在最高分是「同學」時才作用，其他 23 個詞不受影響。
 * 即時模擬（三批 × 兩手 × 三鏡頭）：手放下時 一起→同學 17→9、同學不變；
 * 手停在畫面時 一起→同學 18→0、同學 108→95（示範者 A 的比法手形不明顯）。
 * 門檻比較過一版較嚴的（小指>1.4、其餘<1.3、拇指>0.65、≥3 幀）：手停畫面時同學 108→77，所以採用下面這組。
 * 做法：比到一半（非放下手補判）時，最高分是「同學」但窗格裡還沒看到這個手形，就不累積連續次數，
 * 等看到手形、或手放下用整段補判時再決定。放下手補判的規則不變。
 * 對照 9/25 沒過回歸的「同學只能在放下手時記下」：那一版手停在畫面裡時同學全滅，這一版看得到手形就照常記。
 */
const GATE_MIN_FRAMES = 2;
const Y_PINKY_MIN = 1.3, Y_CURLED_MAX = 1.4, Y_THUMB_MIN = 0.55;

/** 一隻手是不是「大拇指與小指伸直、其餘彎曲」。f 是 258 維原始幀，off 是這隻手的起始欄位。 */
function isThumbPinkyHand(f, off, scaleY) {
  let any = false;
  for (let i = 0; i < 63; i += 1) if (f[off + i] !== 0) { any = true; break; }
  if (!any) return false;
  const px = i => f[off + i * 3], py = i => f[off + i * 3 + 1] * scaleY;
  const d = (a, b) => Math.hypot(px(a) - px(b), py(a) - py(b));
  const ext = (tip, mcp) => d(tip, 0) / Math.max(d(mcp, 0), 1e-6);
  const palm = Math.max(d(5, 0), 1e-6);
  return ext(20, 17) > Y_PINKY_MIN && ext(8, 5) < Y_CURLED_MAX && ext(12, 9) < Y_CURLED_MAX &&
         ext(16, 13) < Y_CURLED_MAX + 0.05 && d(4, 5) / palm > Y_THUMB_MIN;
}

export class ContinuousRecognition {
  constructor(options = {}) {
    this.options = Object.freeze({
      windowMs: 2500,
      maxFrames: 80,
      minFrames: 12,
      minSpanMs: 750,
      resetAfterMs: 700,
      maxInvalidRatio: 0.5,
      confirmations: 3,
      confirmationSpanMs: 450,
      // 食指尖在窗格內走過的範圍（畫面比例，x 與 y 取大者）。
      // 0.06 這個值是量出來的，不是猜的：真實手語片段 407 個窗格裡，
      // 最低的 0.1% 是 0.0809；而把同一張影格複製並加上 σ=0.010 的
      // 雜訊（遠大於 MediaPipe 實際抖動）只有 0.0567。
      // 取 0.06 幾乎不會誤擋真的手語，又擋得住停在原地的手。
      minTipTravel: 0.06,
      /* 慢速裝置修法一：「至少 minFrames 張」在時間上會隨裝置變——每秒 20 格是 600 毫秒，
         每秒 8 格卻要 1,500 毫秒，而一個手語動作的手可見時間中位數只有 1,200–1,275 毫秒。
         於是慢手機上這一關在數學上永遠不成立：連「放下手補判」都拿不到輸入
         （makeInput 回 null，實測每秒 8 格時 100% 是這個原因，有效張數平均 8.3／門檻 12）。
         改成「至少這麼久的有效畫面」，依實測影格間隔換算張數。

         600 不是掃出來的，是反推的：每秒 20 格（每幀 50 毫秒）時 ceil(600/50) = 12，
         與原本的 minFrames 同值。而且每幀 75 毫秒（約每秒 13.3 格）以上時，
         真正綁住的是 minSpanMs（750 毫秒的有效畫面跨度，它要 11 張以上），不是張數，
         所以那個範圍內這一行沒有影響——除非動作中間有追丟（張數少、跨度還夠），
         那時候放寬確實會讓一些原本卡住的窗格過關。實測（1,575 支）：
         每秒 20 格逐支完全相同、每秒 15 格只有 1 支不同（而且是變好）。

         放寬之後綁住的那一關就是 minSpanMs——一個與裝置速度無關的門檻。 */
      minFramesMs: 600,
      // 確認一個詞之後，至少隔這麼久才允許確認<下一個>詞。
      // 這只擋兩個詞的交界，不負責擋「同一個詞被記兩次」——見 releaseMs。
      confirmCooldownMs: 700,
      // 同一個詞要再記一次，必須先觀察到這麼久的「放開」：
      // 手離開畫面、停住不動、信心掉下來、或最高分換成別的詞。
      //
      // 原本這裡是純時間冷卻，結果一個手語動作本身就超過冷卻時間，
      // 冷卻一到，同一個動作的尾巴又被判一次——畫面上每個詞都出現兩次。
      // 「等多久」本來就分不出「同一個動作的延續」與「刻意再比一次」，
      // 兩者的差別是中間有沒有放開，不是間隔多長。
      releaseMs: 400,
      // 手放下的那一刻，窗格裡剛好是整個動作<最完整>的一段——離線整段辨識
      // 94% 就是拿這種窗格算出來的。app.js 會在手離開畫面之後、窗格還沒過期
      // 之前補推論一次，並帶 released = true 進來；那一次只需要這麼多次同一個詞。
      //
      // 為什麼不是照原本的「連續三次」：手已經離開畫面，之後不會再有新的影格，
      // 窗格內容固定不動，再推論幾次都是同一份證據數幾遍，湊出來的「連續三次」
      // 不代表多看了什麼。反過來，堅持要連續三次就等於「手放下之前一定要湊滿」，
      // 而實測考卷 D 裡手可見時間短於 1,250 毫秒的 68 段，100% 沒反應。
      // 真瀏覽器實測（網站的 25 支示範影片 × 兩手 × 3 次 × 兩輪）更直接：87 次「沒反應」
      // 裡有 46 次（53%）正確詞已經 ≥0.90 連續<兩>格，就只差第三格。
      releaseConfirmations: 1,
      // 判斷「手停下來了」只看最近這段時間，不看整個窗格：
      // 窗格有 2.5 秒，裡面還留著上一個動作的尾巴，
      // 用整段去算的話，比完停一下再比第二次永遠測不到停頓。
      recentStillMs: 500,
      ...options,
    });
    const o = this.options;
    for (const key of ['windowMs', 'minSpanMs', 'resetAfterMs', 'confirmationSpanMs', 'confirmCooldownMs']) {
      if (!Number.isFinite(o[key]) || o[key] < 0 || (['windowMs', 'resetAfterMs'].includes(key) && o[key] === 0)) {
        throw new RangeError(`${key} must be a finite, valid millisecond duration`);
      }
    }
    if (!Number.isInteger(o.maxFrames) || o.maxFrames < 1 || o.maxFrames > 80 ||
        !Number.isInteger(o.minFrames) || o.minFrames < 1 || o.minFrames > o.maxFrames ||
        !Number.isInteger(o.confirmations) || o.confirmations < 1 ||
        !Number.isInteger(o.releaseConfirmations) || o.releaseConfirmations < 1 ||
        !Number.isFinite(o.maxInvalidRatio) || o.maxInvalidRatio < 0 || o.maxInvalidRatio > 1 ||
        !Number.isFinite(o.minTipTravel) || o.minTipTravel < 0 ||
        !Number.isFinite(o.minFramesMs) || o.minFramesMs <= 0 ||
        !Number.isFinite(o.releaseMs) || o.releaseMs < 0 ||
        !Number.isFinite(o.recentStillMs) || o.recentStillMs <= 0 ||
        o.minSpanMs > o.windowMs) {
      throw new RangeError('Invalid frame, duration, ratio, or confirmation options');
    }
    this._generation = 0;
    // 「同學」在 labels.json 裡的編號（目前是 6）；app.js 載入 labels 後會用 setHandshapeGate 再設一次。
    this._gateIndex = 6;
    this._lastTime = null;
    // 影格間隔的取樣<不能>放在 _frames 裡。慢裝置上窗格會被停擺偵測清空，
    // 清空後又湊不滿 3 幀去估間隔，於是「要估間隔得先有歷史、
    // 要有歷史得先別被清掉」互相卡死，永遠停在 insufficient_valid_frames。
    // 這個陣列只有 reset() 會清，_clear() 不碰。
    this._gaps = [];
    this._gapPrevTime = null;
    this._clear();
  }

  get generation() {
    return this._generation;
  }

  /** 指定哪一個詞要先看到「大拇指與小指伸直」的手形才能在比到一半時記下；null＝關閉。 */
  setHandshapeGate(index) {
    this._gateIndex = Number.isInteger(index) && index >= 0 && index < 25 ? index : null;
  }

  /** 窗格裡有沒有至少 GATE_MIN_FRAMES 張影格出現這個手形（任一隻手）。 */
  _handshapeSeen() {
    const scaleY = frameAspectScaleY();
    let n = 0;
    for (const item of this._frames) {
      if (!item.frame) continue;
      if (isThumbPinkyHand(item.frame, HAND_OFFSETS[0], scaleY) || isThumbPinkyHand(item.frame, HAND_OFFSETS[1], scaleY)) {
        n += 1;
        if (n >= GATE_MIN_FRAMES) return true;
      }
    }
    return false;
  }

  /** 開始新的一段擷取；還在進行中的推論結果一律作廢。 */
  reset() {
    this._clear();
    this._lastTime = null;
    this._gaps = [];
    this._gapPrevTime = null;
    this._generation += 1;
  }

  _clear() {
    this._frames = [];
    this._expiryHoldUntil = null;
    this._missingSince = null;
    this._lastPushAt = null;
    this._candidate = null;
    this._cooldownUntil = null;
    this._lastConfirmedIndex = null;
    this._releaseSince = null;
  }

  _time(now) {
    if (!Number.isFinite(now) || now < 0) throw new RangeError('Time must be a finite, nonnegative millisecond value');
    if (this._lastTime !== null && now < this._lastTime) throw new RangeError('Times must be monotonic');
    this._lastTime = now;
  }

  /**
   * 實際影格間隔的中位數。用中位數而不是平均，因為偶爾一張特別慢的影格
   * （瀏覽器排程、垃圾回收）不該把整個估計拉走。
   */
  _frameIntervalMs() {
    if (this._gaps.length < 3) return null;
    const gaps = this._gaps.slice().sort((a, b) => a - b);
    return gaps[gaps.length >> 1];
  }

  /** 每次 push 都記一筆間隔，最多留最近 20 筆。 */
  _recordGap(time) {
    if (this._gapPrevTime !== null) {
      const gap = time - this._gapPrevTime;
      if (gap > 0 && gap < 5000) {
        this._gaps.push(gap);
        if (this._gaps.length > 20) this._gaps.shift();
      }
    }
    this._gapPrevTime = time;
  }

  /**
   * 這台裝置在這個窗格裡放得下幾張影格？據此放寬 minFrames。
   * 慢裝置不該因為湊不滿 12 張就永遠無法辨識——那是條件本身不可能成立，
   * 不是使用者比錯。
   */
  _effectiveMinFrames() {
    const gap = this._frameIntervalMs();
    if (gap === null || gap <= 0) return this.options.minFrames;
    const capacity = Math.floor((this.options.windowMs / gap) * 0.6);
    // minFramesMs 的有效畫面在這台裝置上是幾張（每秒 20 格 → 12，8 格 → 5）。
    const byTime = Math.ceil(this.options.minFramesMs / gap - 1e-9);
    // 下限訂 3：每秒一幀的裝置在 2.5 秒窗格裡也只裝得下 2 到 3 幀，
    // 下限訂得比容量高，條件就永遠不可能成立。
    return Math.min(this.options.minFrames, Math.max(3, Math.min(byTime, capacity)));
  }

  /** 慢裝置的影格本來就疏，固定 700 毫秒會把正常的間隔誤判成「來源停擺」。 */
  _effectiveResetAfterMs() {
    const gap = this._frameIntervalMs();
    // 還沒估出間隔時（開頭前幾幀）給一段寬限，不然慢裝置在拿到第一個
    // 估計值之前就已經被清空好幾輪了。
    if (gap === null || gap <= 0) return Math.max(this.options.resetAfterMs, 1500);
    return Math.max(this.options.resetAfterMs, gap * 3);
  }

  /**
   * 食指尖在窗格內走過的範圍（x 與 y 的極差取大者，兩隻手取大者）。
   * 用「走過多遠」而不是「每幀移動多少」，是因為停在原地的手會抖，
   * 逐幀位移抓得到抖動、抓不到「其實沒去任何地方」。
   */
  _tipTravel(since = null) {
    let best = 0, judged = false;
    for (const off of HAND_OFFSETS) {
      let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, seen = 0;
      for (const item of this._frames) {
        if (!item.valid) continue;
        if (since !== null && item.time < since) continue;
        const base = off + INDEX_TIP * 3;
        const x = item.frame[base], y = item.frame[base + 1];
        // 手沒被偵測到時整塊是零；(0,0) 不是真的位置，不能算進極差。
        if (x === 0 && y === 0) continue;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        seen += 1;
      }
      if (seen >= 3) { best = Math.max(best, maxX - minX, maxY - minY); judged = true; }
    }
    // 判斷不出來就回 null，代表「不知道」，上層要放行而不是擋下來。
    // 這一關的目的是擋掉「停著不動」，不是擋掉「資料太少無法判斷」——
    // fail-closed 會讓慢裝置永遠什麼都顯示不出來，比偶爾多一個字嚴重得多。
    return judged ? best : null;
  }

  /**
   * 記錄「有沒有在放開」。連續放開滿 releaseMs 就把 _lastConfirmedIndex 清掉，
   * 那個詞才能再被記一次。中途只要又出現「同一個詞、夠有信心」就重新計時。
   */
  _noteRelease(now, releasing) {
    if (this._lastConfirmedIndex === null) return;
    if (!releasing) { this._releaseSince = null; return; }
    if (this._releaseSince === null) { this._releaseSince = now; return; }
    if (now - this._releaseSince >= this.options.releaseMs) {
      this._lastConfirmedIndex = null;
      this._releaseSince = null;
      // 放開＝上一個動作真的結束了，它留在窗格裡的影格也要丟掉。
      // 不丟的話，那些影格會被當成新動作再判一次——同一個詞還是出現兩次，
      // 只是從「冷卻一到就重複」變成「放開一到就重複」。
      this._frames = [];
      this._candidate = null;
      this._missingSince = null;
    }
  }

  /** 最近 recentStillMs 之內，手有沒有幾乎沒移動。資料不足以判斷時回 false（不算放開）。 */
  _recentlyStill(now) {
    const travel = this._tipTravel(now - this.options.recentStillMs);
    return travel !== null && travel < this.options.minTipTravel;
  }

  _advance(now) {
    this._time(now);
    // 手明確不見、或影格來源暫停／卡住，兩種情況都要讓窗格過期。
    const limit = this._effectiveResetAfterMs();
    const missingExpired = this._missingSince !== null && now - this._missingSince >= limit;
    const captureStalled = this._lastPushAt !== null && now - this._lastPushAt >= limit;
    let reset = false;
    /* 慢速裝置修法二：補判（放下手那一次）還在飛的時候先不要過期。
       過期會推進 _generation，而 app.js 的 predictLatest 一看到世代變了就把整個
       結果丟掉——包含已經算對的那個詞。期限原本是「700 毫秒 − 推論時間」，
       所以推論一慢（手機約 400 毫秒）補判就一定來不及。
       這裡只是把那段等待讓出來：補判一回來 app.js 立刻解除（見 holdExpiry），
       之後下一張影格照樣會過期。所以「補判本來就來得及」的裝置行為完全不變。 */
    const holdingForRelease = this._expiryHoldUntil !== null && now < this._expiryHoldUntil;
    if ((missingExpired || captureStalled) && !holdingForRelease) {
      this._clear();
      this._generation += 1;
      reset = true;
    }
    const cutoff = now - this.options.windowMs;
    while (this._frames.length && this._frames[0].time < cutoff) this._frames.shift();
    return reset;
  }

  _state(reset = false, now = null, released = false) {
    const valid = this._frames.filter(item => item.valid);
    const totalFrames = this._frames.length;
    const validFrames = valid.length;
    const invalidRatio = totalFrames ? (totalFrames - validFrames) / totalFrames : 1;
    const validSpanMs = validFrames > 1 ? valid.at(-1).time - valid[0].time : 0;
    const minFrames = this._effectiveMinFrames();
    const travel = this._tipTravel();
    let reason = null;
    if (!totalFrames) reason = 'empty';
    else if (validFrames < minFrames) reason = 'insufficient_valid_frames';
    else if (validSpanMs < this.options.minSpanMs) reason = 'insufficient_span';
    // released（手已經放下）時不套這一條。這道關卡問的是「這個人現在是不是在比」，
    // 手放下之後問它沒有意義；而且 preprocess.resampleActive 只把窗格裡<有手>的那一段
    // 重取樣送進模型，沒手的影格根本沒被餵進去。
    // 實測（驗證集 838 段）：等 250 毫秒時有 40–45 次補判被這一條擋掉，其中 25–32 次
    // 模型其實已經正確且 ≥0.90——放寬之後筆電 4:3 正確率 74.0%→81.1%，而認錯與多出字
    // 一個都沒有增加（3.6%／0.5%，兩邊相同）。minSpanMs、minFrames、動作量門檻都還在。
    else if (invalidRatio > this.options.maxInvalidRatio && !released) reason = 'too_many_invalid_frames';
    // travel 為 null＝資料不足以判斷，放行。
    else if (travel !== null && travel < this.options.minTipTravel) reason = 'not_moving';
    return {
      reset, ready: reason === null, reason, validFrames, totalFrames, validSpanMs,
      invalidRatio, travel, minFrames, generation: this._generation,
      frameIntervalMs: this._frameIntervalMs(),
      coolingDown: now !== null && this._cooldownUntil !== null && now < this._cooldownUntil,
    };
  }

  /**
   * 加入一幀 258 維特徵。無效的幀以全零列表示，傳進來的數值不會進到模型。
   * 輸入會複製一份，不保留原本的參照。
   */
  push(frame, valid, time) {
    if (typeof valid !== 'boolean') throw new TypeError('valid must be a boolean');
    if (valid && (!(frame instanceof Float32Array) || frame.length !== 258)) {
      throw new TypeError('A valid frame must be Float32Array(258)');
    }
    if (valid && !frame.every(Number.isFinite)) throw new TypeError('Landmarks must be finite');
    this._recordGap(time);
    const reset = this._advance(time);
    if (valid) this._missingSince = null;
    else if (this._missingSince === null) this._missingSince = time;
    this._lastPushAt = time;
    // 有拿到 258 維就存，不看 valid。訓練片段裡「手沒被偵測到」的幀是
    // 手部欄位為零、姿態還在，它們也在重取樣的區間內；瀏覽器要一樣。
    const keep = (frame instanceof Float32Array && frame.length === RAW_DIM) ? frame.slice() : null;
    this._frames.push({ frame: keep, valid, time });
    if (this._frames.length > this.options.maxFrames) this._frames.splice(0, this._frames.length - this.options.maxFrames);
    const state = this._state(reset, time);
    // 放開只認<正向訊號>：手不在畫面、或手停住不動。
    // 不能用「窗格還沒準備好」——確認一個詞之後窗格本來就被清空，
    // 那個空窗會被誤判成放開，於是同一個動作的尾巴又被記一次（就是那個「每個詞出現兩次」）。
    // 「資料還不夠」是還在累積，不是使用者放開了。
    this._noteRelease(time, !valid || this._recentlyStill(time));
    return state;
  }

  /**
   * 回傳模型輸入 Float32Array(80*291)，或 null（窗格還沒準備好）。
   *
   * 前處理完全交給 preprocess.buildModelInput，那是照 Python 的
   * training/tslkit.py 的 prep() 逐行搬的：抓有手區間重取樣 80 幀 → 指向幾何 33 維 →
   * 雙肩標準化 → 串接成 291 維。瀏覽器這邊不自己做任何取樣，
   * 否則餵進模型的格式就跟訓練時不一樣——實測證明過那會明顯掉分。
   */
  makeInput(now, withMirror = false, released = false) {
    this._advance(now);
    if (!this._state(false, now, released).ready) return null;
    const frames = [];
    for (const item of this._frames) if (item.frame) frames.push(item.frame);
    if (!frames.length) return null;
    const plain = buildModelInput(frames);
    if (!withMirror) return plain;
    // 測試時鏡像：同一段動作加做一份左右鏡像，兩個視角的機率平均。
    // 不用重新訓練，實測在難的考卷上 +3.3、容易的考卷上不變，從來不會變差。
    const mirrored = buildMirroredInput(frames);
    const pair = new Float32Array(plain.length * 2);
    pair.set(plain, 0); pair.set(mirrored, plain.length);
    return pair;
  }

  /**
   * 慢速裝置修法二：在 until（毫秒，與 push 的時間軸同一個）之前不要讓窗格過期。
   * 呼叫端＝app.js：送出補判時設 now + RELEASE_EXPIRY_HOLD_MS，補判回來時傳 0 解除。
   * 傳 0／null／非數字都是解除。上限由呼叫端給，這裡不自己訂，避免兩邊各有一個期限。
   */
  holdExpiry(until) {
    this._expiryHoldUntil = (Number.isFinite(until) && until > 0) ? until : null;
  }

  /** 只查詢窗格是否準備好，不產生模型輸入。 */
  getState(now) {
    const reset = this._advance(now);
    return this._state(reset, now);
  }

  /**
   * 判斷一次完成的推論結果。要依序呼叫（同時只有一個推論在跑）。
   * 傳入呼叫 makeInput 當時的 generation，重設之前送出的非同步結果才會被丟掉。
   *
   * 確認一個詞之後會把窗格清空並進入冷卻。清空是為了讓下一個詞從乾淨的
   * 影格重新累積（否則兩個詞的交界會被判成第三個詞）；冷卻取代了原本
   * 「同一個詞永遠不能再記錄一次」的封鎖，所以「我再比一次」現在會成立。
   *
   * released＝呼叫端判定「手已經放下、這個動作結束了」（見 app.js 的 onResults）。
   * 那一次用 releaseConfirmations 取代連續次數與 confirmationSpanMs。
   */
  observe(scores, threshold, now, generation = this._generation, released = false) {
    if (!Number.isFinite(threshold) || threshold < 0 || threshold > 1) throw new RangeError('Threshold must be between 0 and 1');
    this._advance(now);
    const base = { top: null, qualified: false, stable: false, confirmed: null, generation: this._generation };
    if (generation !== this._generation) return { ...base, reason: 'stale_generation' };
    const state = this._state(false, now, released);
    if (!state.ready) {
      this._candidate = null;
      this._noteRelease(now, state.reason === 'not_moving' || this._recentlyStill(now));
      return { ...base, reason: state.reason === 'not_moving' ? 'not_moving' : 'not_ready', state };
    }
    if (!scores || scores.length !== 25 || !Array.from(scores).every(value => Number.isFinite(value) && value >= 0 && value <= 1)) {
      this._candidate = null;
      return { ...base, reason: 'invalid_scores' };
    }
    let index = 0;
    for (let i = 1; i < scores.length; i += 1) if (scores[i] > scores[index]) index = i;
    const top = { index, score: scores[index] };
    if (top.score < threshold) {
      this._candidate = null;
      this._noteRelease(now, true);
      return { ...base, top, reason: 'low_confidence' };
    }
    // 最高分換成別的詞，代表剛才那個詞已經結束了。
    // 但「最高分還是同一個詞」不可以反過來取消放開計時——比完停在原地時，
    // 模型仍然會很有信心地報同一個詞（它沒有「沒在比」這一類），
    // 那會把 push() 記到的停頓抹掉，於是停多久都沒辦法再記一次。
    this._noteRelease(now, index !== this._lastConfirmedIndex || this._recentlyStill(now));
    // 「同學」手形關卡（見檔頭 GATE_MIN_FRAMES 的說明）：比到一半、還沒看到手形，就先不累積。
    if (!released && this._gateIndex !== null && index === this._gateIndex && !this._handshapeSeen()) {
      this._candidate = null;
      return { ...base, top, qualified: true, stable: false, confirmed: null, reason: 'awaiting_handshape',
               consecutive: 0, stableSpanMs: 0, generation: this._generation };
    }
    if (!this._candidate || this._candidate.index !== index) {
      this._candidate = { index, count: 1, since: now };
    } else {
      this._candidate.count += 1;
    }
    const stable = released
      ? this._candidate.count >= this.options.releaseConfirmations
      : this._candidate.count >= this.options.confirmations && now - this._candidate.since >= this.options.confirmationSpanMs;
    const cooling = this._cooldownUntil !== null && now < this._cooldownUntil;
    // 同一個詞在放開之前不能再記一次——否則一個動作的尾巴會被判成第二次。
    const blocked = this._lastConfirmedIndex === index;
    let confirmed = null;
    if (stable && !cooling && !blocked) {
      confirmed = { index, score: top.score, time: now };
      // 清掉窗格並進入冷卻：下一個詞從零開始累積，不會摻到這一個詞的尾巴。
      this._frames = [];
      this._candidate = null;
      this._missingSince = null;
      this._cooldownUntil = now + this.options.confirmCooldownMs;
      this._lastConfirmedIndex = index;
      this._releaseSince = null;
      // 這裡<不可以>推進 _generation。呼叫端（app.js 的 predictLatest）在
      // observe 回來之後會比對 `recognitionGeneration !== continuous.generation`，
      // 一旦不相等就整個結果丟掉——包含剛確認的那個詞。
      // 結果是每個詞都確認成功、每個詞都在下一行被丟掉，畫面上什麼都不會出現。
      // 清空窗格已經足夠讓還在飛的推論失效：它回來時窗格是空的，
      // _state().ready 為 false，會走 not_ready 而不會誤確認。
    }
    return {
      ...base, top, qualified: true, stable, confirmed,
      reason: confirmed ? 'confirmed'
        : (blocked && stable) ? 'awaiting_release'
        : cooling ? 'cooling_down'
        : stable ? 'stable' : 'stabilizing',
      consecutive: this._candidate ? this._candidate.count : 0,
      stableSpanMs: this._candidate ? now - this._candidate.since : 0,
      generation: this._generation,
    };
  }
}
