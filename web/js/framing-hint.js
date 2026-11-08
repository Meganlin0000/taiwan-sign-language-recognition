/**
 * 位置提示：只在使用者「沒在比」的時候，用一句話說該怎麼調整鏡頭。
 *
 * 量什麼（只用 MediaPipe 姿態的左右肩，第 11、12 點）
 *   swPx = hypot((x11-x12)*W, (y11-y12)*H)   W、H = 送進 MediaPipe 的畫布像素（app.js 的 inputCanvas，寬固定 640）
 *   寬度  = W / swPx                 畫面寬是幾個肩寬（越小＝人越滿、越近）
 *   底部  = (1-cy)*H / swPx          肩線到畫面底有幾個肩寬
 *   左右  = min(cx, 1-cx)*W / swPx   身體中線到較近那一側
 * 先乘回像素再除以「像素肩寬」，所以同一組門檻在 16:9、4:3、3:4 都成立。
 *
 * 門檻怎麼定的：兩個條件都要成立才留下這一項提示。
 *   (a) 不可以打到錄製資料本身的取景。四位示範者全部 7,167 支片段（含考卷 D 的 343 支）
 *       每一位單獨看都要 ≤2% 才會被提醒——錄的時候就是這樣坐、而且辨識得出來的取景，沒有資格被糾正。
 *   (b) 要有理由：量到的準確率差異，或「手根本不會被看到」這種機制上的必然。
 *
 *   ⚠ 這一版把前一版的門檻放寬了。前一版說「門檻＝全部片段第 1～2 百分位」，
 *   但重算後不是：太近 3.0 個肩寬會打到全部片段的 15.4%（A 的 4:3 18.8%、B 12.7%）、
 *   太遠 110 像素會打到 C 15.8%、偏一邊 0.8 會打到 B 7.9%、頭頂 0.55 會打到 B 7.3%。
 *   作者平常的取景（4:3、肩寬占畫面寬 34%＝2.94 個肩寬）在前一版會 99% 的時間掛著「請往後坐一點」。
 *   同時刪掉「頭頂被切」：整段辨識網格顯示肩線高度 45%～72% 準確率幾乎一樣，頭被切到並沒有量到壞處，
 *   而示範者 B 也有 5～10% 的片段就是頭切掉的（肩線在畫面高 10～26%）——沒有依據的提醒不留。
 */

/** 門檻。每一組都是（出現, 收起）兩個值，中間維持原狀＝遲滯，站在邊界上不會一直換字。 */
export const FRAMING = Object.freeze({
  // 太近：畫面寬不到 2.5 個肩寬（4:3 下＝肩寬占畫面寬 40%）。全部片段 1.9%，每位示範者 0～2.6%。
  // 有量到：即時模擬（驗證集 16:9 重投影成 4:3）2.50 個肩寬 51.7%、3.57 個肩寬 60.5%，差 8.8 個百分點。
  TOO_CLOSE: 2.5, CLOSE_OK: 2.7,
  // 太遠：肩寬不到 90 像素（640 寬畫布）。全部片段 0.1%（最遠的 C 1.7%）。
  // 這麼遠時整隻手只有 20 幾像素，MediaPipe 常常整段抓不到手 → 不是準確率變差，是完全沒反應。
  TOO_FAR_PX: 90, FAR_OK_PX: 100,
  // 胸口被切：肩線到畫面底不到 0.40 個肩寬。全部片段 1.1%，每位 0～1.3%。
  // 手的核心範圍在肩線下方 0.59 個肩寬（全部片段的中位數），所以低於 0.40 時手整個在畫面外，
  // 而網站只要看不到手就完全不推論（validFraming）→ 一直「沒反應」。
  BOTTOM_MIN: 0.40, BOTTOM_OK: 0.50,
  // 偏一邊：中線到較近一側不到 0.6 個肩寬（＝那一邊的肩膀離畫面邊只剩 0.1 個肩寬）。每位示範者 0～1.7%。
  // 手的最大水平伸展中位數 1.17 個肩寬，所以偏到這個程度時，往那一邊比的手會整隻出框（同上，出框就停止推論）。
  // 整段辨識只測過比較輕微的偏移（中線移到畫面 70% 處：驗證集 97.8%→95.9%），所以「偏多少會掉多少」沒有數字。
  SIDE_MIN: 0.6, SIDE_OK: 0.75,
  // 稍微近了（只是建議）：3.1 個肩寬（4:3 ＝ 肩寬 32%）。作者平常的取景（2.94 個肩寬）就落在這裡。
  // 依據是成對比較 肩寬 34%→28%：驗證集 19 支變對、5 支變錯（p=0.007），D 26 比 7（p=0.001），約 +3 個百分點。
  // 會打到全部片段的 22%，所以只能當建議、不能常駐（見 app.js 的 framingCoaching）。
  A_BIT_CLOSE: 3.1, A_BIT_OK: 3.3,

  // 註：沒有「肩膀超出畫面就算太近」這條規則。試過，但人很小的時候 MediaPipe 會把肩膀外推到畫面外，
  // 於是「人在很遠」被判成「太近」（端對端實測：畫面寬 5.6 個肩寬卻顯示「請往後坐一點」）。
  // 真的太近時 width 本來就會小於門檻、偏一邊時 side 會小於門檻，這條規則沒有補到別的情況，所以刪掉。
  SHOULDER_VIS: 0.5,   // 肩膀可見度低於此＝座標是 MediaPipe 推出來的 → 只判斷距離，不判斷「偏一邊／胸口被切」
  WINDOW_MS: 1000,     // 取最近這段時間的中位數，不看單一幀
  MIN_SAMPLES: 5,      // 視窗內至少這麼多幀才判斷（20 fps ≈ 0.25 秒）；手放下的那一瞬間不算
  SHOW_AFTER_MS: 1000, // 同一個問題連續這麼久才顯示
  HIDE_AFTER_MS: 1000, // 問題消失這麼久才收起
});

/** 「擋路的」＝辨識幾乎做不起來，隨時可以提醒。其餘（a_bit_close、portrait）是建議，由 app.js 決定何時講。 */
// 'portrait' 在三個修法合併時從這裡拿掉了（長寬比修法讓直拿不再變差），理由寫在 _raw() 裡。
export const HARD = Object.freeze(['no_person', 'too_close', 'too_far', 'chest_cut', 'off_center']);
export const isHard = code => HARD.includes(code);

/** 顯示在辨識卡片上的字。短、只講下一步做什麼。有「請」＝真的擋路；沒有「請」＝建議。 */
export const FRAMING_TEXT = Object.freeze({
  no_person:   '請坐到鏡頭前',
  portrait:    '手機橫著拿會更好比',   // 建議（沒有「請」）：直拿已經不會比較不準，只是手比較容易比出畫面
  too_close:   '請往後坐一點',
  too_far:     '請坐近一點',
  chest_cut:   '請讓胸口也入鏡',
  off_center:  '請坐到畫面中間',
  a_bit_close: '往後坐一點會更準',
});

/** 一幀的量測。沒有人、或量不出肩寬時回 null。 */
export function measure(pose, W, H) {
  if (!pose || pose.length !== 33) return null;
  const l = pose[11], r = pose[12];
  if (!l || !r) return null;
  const swPx = Math.hypot((l.x - r.x) * W, (l.y - r.y) * H);
  if (!(swPx > 1)) return null;
  const cx = (l.x + r.x) / 2, cy = (l.y + r.y) / 2;
  return {
    swPx,
    width: W / swPx,
    bottom: (1 - cy) * H / swPx,
    side: Math.min(cx, 1 - cx) * W / swPx,
    vis: Math.min(l.visibility ?? 1, r.visibility ?? 1),
  };
}

const median = a => { const s = a.slice().sort((p, q) => p - q); return s[s.length >> 1]; };

/**
 * 逐幀呼叫 update()，回傳目前該顯示的問題代碼或 null。
 * 同一個問題要連續 SHOW_AFTER_MS 才出現；解除要連續 HIDE_AFTER_MS 才收起；一次只回一個代碼。
 */
export class FramingMonitor {
  constructor(opt = FRAMING) { this.o = opt; this.reset(); }
  reset() { this.buf = []; this.shown = null; this.cand = null; this.candSince = 0; this.okSince = null; }

  _raw(now, portrait) {
    const o = this.o;
    const recent = this.buf.filter(b => now - b.t <= o.WINDOW_MS);
    if (recent.length < o.MIN_SAMPLES) return this.shown;      // 取樣不夠（剛放下手、正在比）：維持原狀
    const people = recent.filter(b => b.m);
    if (people.length < recent.length / 2) return 'no_person';
    const m = k => median(people.map(b => b.m[k]));
    const was = this.shown;
    /* 距離（太近 → 太遠 → 稍微近了）只看「像素肩寬」，這是最禁得起雜訊的一個量：
       網站的畫布寬固定 640，所以 width = 640 / swPx，兩個門檻其實是同一條軸的兩端
       （2.5 個肩寬 = 256 像素、90 像素 = 7.1 個肩寬）。人很小或側身時肩膀可見度會掉，
       但肩膀間距本身還是可信的，所以這一段不受可見度影響。 */
    const width = m('width'), swPx = m('swPx');
    if (width < (was === 'too_close' ? o.CLOSE_OK : o.TOO_CLOSE)) return 'too_close';
    if (swPx < (was === 'too_far' ? o.FAR_OK_PX : o.TOO_FAR_PX)) return 'too_far';
    /* 「人在畫面的哪裡」對座標精度比較敏感。肩膀可見度低（人很小、側身、被擋住）時
       MediaPipe 是用推的，這時不判斷位置——但也不要繼續掛著上一句（那會變成用不可信的資料
       堅持一個可能已經不對的提示，端對端實測就遇過：人退得很遠、卻還寫著「請往後坐一點」）。 */
    if (m('vis') >= o.SHOULDER_VIS) {
      if (m('bottom') < (was === 'chest_cut' ? o.BOTTOM_OK : o.BOTTOM_MIN)) return 'chest_cut';
      if (m('side') < (was === 'off_center' ? o.SIDE_OK : o.SIDE_MIN)) return 'off_center';
    }
    /* 手機直拿。<這一段在三個修法合併時改過>，原因寫在這裡，因為它是唯一一處
       「另一個修法把依據拿掉了」的地方。

       原本：直拿是第一順位、而且是「擋路」等級，`if (portrait) return 'portrait'` 擺在
       整個函式最前面。依據是即時模擬的正確率——直拿 3:4 只有 37%，筆電 4:3 55%、原片 16:9 66%，
       認錯從 2% 跳到 22%。那個差距的成因是前處理的縱向尺度乘上「畫面寬÷高」（3:4 = 0.75）。

       現在：長寬比修法（preprocess.js 的 setFrameAspect）已經在送進模型前把 y 換算回 16:9，
       所以那個成因不存在了。實測（三個修法合併、手放下、兩手合計）：
         考卷 D 直拿 88.2%／認錯 4.4%，筆電 4:3 86.3%／認錯 4.7%，原片 16:9 88.0%／認錯 4.5%
         驗證集 直拿 88.4%／認錯 0.8%，筆電 4:3 84.8%／認錯 1.0%
       直拿現在<不比橫拿差>（模擬裡甚至略好，因為橫拿的 4:3 是裁掉左右、真的少看到東西）。
       繼續叫人把手機轉過來，等於建議使用者換到比較差的那一邊。

       所以改成兩點：
       (1) 不再是「擋路」等級（見下面的 HARD），只在取景階段講一次，講完就不再插嘴。
       (2) 移到所有擋路判斷<之後>。原本擺在最前面會短路掉全部其他判斷——實測直拿時
           這一句 8/8 永遠掛著，使用者就算沒入鏡、坐太近、胸口被切，也只會看到「請把手機橫著拿」。

       為什麼不是整條刪掉：真手機直拿的水平視角比橫拿窄，手往兩側比比較容易出畫面，而網站
       只要看不到手就完全不推論。這一點這套離線模擬與 e2e 都測不到（兩邊的直拿都是把 16:9
       上下補黑邊做出來的，人在畫面裡的大小跟 4:3 一樣）。所以保留這句話，但降級成建議，
       而且文案不再宣稱準確率——真手機上如果人太滿，too_close 那條規則本來就會先講話。 */
    /* 09-26 全面檢查後拿掉 `if (portrait) return 'portrait';`：直拿時這一條永遠成立，
       卡片一直叫人「手機橫著拿會更好比」、目標人形不淡出，看完示範回來又講一次，平板直拿也寫「手機」。
       而長寬比校正之後直拿並不比較差（考卷 D 手機直拿 88.2%、筆電 86.3%）。
       FRAMING_TEXT.portrait 留著沒用到，將來真手機實測證明直拿較差再接回來。 */
    if (width < (was === 'a_bit_close' ? o.A_BIT_OK : o.A_BIT_CLOSE)) return 'a_bit_close';
    return null;
  }

  /**
   * pose   = results.poseLandmarks（沒有人就傳 undefined）
   * W, H   = inputCanvas.width / height
   * sample = 這一幀要不要拿來量位置。只在「手沒在比」時量：
   *          比手語時身體會轉、會前傾，同一支片段的肩寬量起來 2.9～3.4 個肩寬，
   *          拿來判斷距離會讓提示在比的過程中跳來跳去。
   */
  update(pose, W, H, now, sample = true) {
    if (sample) this.buf.push({ t: now, m: measure(pose, W, H, this.o) });
    while (this.buf.length && now - this.buf[0].t > this.o.WINDOW_MS) this.buf.shift();
    const raw = this._raw(now, H > W);
    if (raw === this.shown) { this.cand = null; this.okSince = null; return this.shown; }
    if (raw === null) {                                   // 問題解除：要穩定一段時間才收
      if (this.okSince === null) this.okSince = now;
      if (now - this.okSince >= this.o.HIDE_AFTER_MS) { this.shown = null; this.okSince = null; }
      return this.shown;
    }
    this.okSince = null;
    if (this.cand !== raw) { this.cand = raw; this.candSince = now; }
    if (now - this.candSince >= this.o.SHOW_AFTER_MS) { this.shown = raw; this.cand = null; }
    return this.shown;
  }
}
