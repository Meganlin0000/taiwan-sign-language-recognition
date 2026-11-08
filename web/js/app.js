import {extractKeypoints, validFraming, rankScores} from './recognition-core.js';
import {ContinuousRecognition} from './continuous-core.js';
import {setFrameAspect} from './preprocess.js';
import {FramingMonitor, FRAMING_TEXT, isHard} from './framing-hint.js';
import {Speech} from './speech.js?v=unlock2';

const $ = id => document.getElementById(id);
const video = $('camera'), overlay = $('overlay'), overlayContext = overlay.getContext('2d');
const inputCanvas = document.createElement('canvas'), inputContext = inputCanvas.getContext('2d');

// 辨識出的句子。取代舊版的「紀錄」清單——同一份資料只呈現一次，介面才不會散。
const sentence = [];
const SENTENCE_LIMIT = 40;
/* 手放著、等人比的時候卡片上的大字。原本是「開始比一個詞」；09-25 作者問「要怎麼清楚告知比完手要放下」：
   手放下的那一刻網站會補判一次（最完整的一段動作在那時候），慢手機上兩個詞之間也要停一下。
   講在這裡是因為這句只在「手放著、準備比下一個」時出現，時機剛好；不另外加提示框。 */
const IDLE_PROMPT = '比一個詞，比完手放下';
/* 記下一個詞之後手還舉著、還在動：卡片改說這句（09-25 作者：「每比完一個動作手都要放下，這很重要」）。
   手沒放下就接著比，窗格裡會混到上一個詞的尾巴，最容易認錯。
   手一離開畫面、或手停住不動，就算放下了，換回 IDLE_PROMPT。只改卡片上的字，辨識規則不變。 */
const LOWER_PROMPT = '手放下，再比下一個';
/* 開鏡頭後的頭 5 秒（還沒記下任何詞）卡片先講鏡頭高度（09-25 作者實測：手機、平板放桌上從下往上拍明顯比較不準，
   平視比較準，「這很重要，要顯示在畫面上」）。開鏡頭前的框裡也有同一句。有更具體的位置問題（請往後坐一點…）時先講那個。
   網站看不出鏡頭角度（位置判斷只看肩寬、置中、胸口），所以不能只在「角度不對時」才講，改成每次開鏡頭講一次。 */
const EYE_PROMPT = '鏡頭跟臉\u200b一樣高';   // \u200b：窄的直拿畫面放不下一行時折成「鏡頭跟臉／一樣高」，不要剩一個「高」
const EYE_PROMPT_MS = 5000;
let cameraStartedAt = -Infinity;
const idlePrompt = () => (framingCoaching && performance.now() - cameraStartedAt < EYE_PROMPT_MS ? EYE_PROMPT : IDLE_PROMPT);
let lowerHint = false;

let guide = [], labels = {}, localGuide = null, stream, holistic, worker, preparing;
let active = false, paused = false, generation = 0, lastFrame = 0, loopId = 0;
let frameTask = Promise.resolve(), processingGeneration = -1, processingRecognitionGeneration = -1, requestId = 0;
let fatalRuntime = false, latestFrameValid = false, lastPredictionAt = -Infinity;
/* 手從哪一刻開始不在畫面裡（null = 現在看得到手）。給「放下手補判一次」用。 */
let handsGoneSince = null, releaseDone = false;
let inferenceSerial = 0, inFlightPrediction = 0;
let lastConfirmedAt = -Infinity;
let starting = false;          // 按下「開始辨識」到鏡頭真的開好之間
let stopLockUntil = 0;         // 主按鈕剛變成「停止」的一小段時間內不接受「停止」（防連點）
let vendorFailed = false;       // 辨識元件的檔案下載失敗過（見 trackVendorDownload）：「重新載入」要重新整理頁面
let startLockUntil = 0;        // 剛按過「停止」的一小段時間內不接受「開始辨識」（防連點，反方向）
let speaking = false;          // 正在唸整句   // 上一次真的寫進句子的時間；連續失敗太久要改口給建議
const pending = new Map();
const continuous = new ContinuousRecognition();
/* 位置提示：人不在畫面裡、坐太近、手機直拿時，在辨識卡片上說一句該怎麼調整。
   只在「手沒在比」（手放著、或手停著不動）的時候量、也只在那時候說；正在比的時候不插話。
   門檻與依據寫在 framing-hint.js。 */
const framing = new FramingMonitor();
/* 「建議」等級的提示（往後坐一點會更準、坐到畫面中間…）只在取景階段講：
   從開鏡頭到這一輪第一次成功記下詞為止。成功之後就不再插話，不然每比完一個詞、手一放下
   都會再跳一次同一句，變成整場都在唸使用者（作者平常的取景本來就偏近，會 100% 命中「建議」）。
   之後若連續 6 秒認不出來，會改在狀態列（小字）提一次，見 predictLatest。 */
let framingCoaching = true;
let showTargetPose = true;     // 畫面上的目標骨架（對位用）
let framingIssue = '';         // 目前的取景問題（有問題時目標人形畫得比較亮）
let alignedSince = 0;          // 位置從什麼時候開始是乾淨的（連續對準一段時間，人形才淡出）
const speech = new Speech('voices');
/* 手機、平板只准在「使用者點下去的當下」開始出聲；自動唸出發生時沒人在點，會被擋掉（2026-09-25 實際回報）。
   所以整頁任何一次點擊、按鍵都順手解鎖聲音（按「開始辨識」一定會經過這裡）。解鎖過就不做事，很便宜。 */
for (const type of ['touchend', 'click', 'keydown']) {
  document.addEventListener(type, () => speech.unlock(), {capture: true, passive: true});
}
/* 記進句子所需的最低信心。原本是使用者可調的滑桿，拿掉了——
   實測顯示模型過度自信（上線的四人版考 D：信心 0.93 以上那一格自稱 99.7%、實際 95.6%；
   「你」平均信心 0.985、實際只對 26.7%），所以那個數字擋不掉多少錯誤，調高它也不會讓答案更可靠。
   同一個理由，畫面上不顯示任何信心百分比（2026-09-24 全站檢查）。
   真正在過濾雜訊的是 continuous-core 的「連續數幀都指向同一個詞」那道關卡。
   等模型完成溫度校準之後，這個值要重新量一次。 */
const CONFIRM_THRESHOLD = 0.90;
const INFERENCE_INTERVAL_MS = 250;
/* 手離開畫面多久之後才算「這個詞比完了」。
   不能一離開就算：實測驗證集 838 段裡有 43.7% 在動作<中間>至少掉一次手
   （MediaPipe 追丟，不是人比完了；長度分布以 1–2 幀最多，403 次 1 幀、104 次 2 幀），
   那時候窗格裡只有動作前半段，拿去確認就是「用半個動作下結論」。實測有這種例子：
   「洗臉」在第 1,240 毫秒掉了三幀，只等 150 毫秒就會先記成「聊天」；等 250 毫秒就
   略過那次追丟，正確記成「洗臉」。
   實測（驗證集 838 段 × 筆電 4:3，正確%／多出字%）：
   等 150 毫秒 80.9／1.6、等 250 毫秒 81.1／**0.5**、等 350 毫秒 81.1／0.5。
   原本這裡寫「<不要往上加>：等待時間加上推論時間必須小於 resetAfterMs（700 毫秒）」。
   那個限制已經由 RELEASE_EXPIRY_HOLD_MS 解掉（補判飛行期間窗格不過期），所以
   250 現在不是被期限逼出來的，而是上面那組「多出字」的數字選出來的：
   等 150 毫秒會把動作中間的追丟誤判成比完了。往上加仍然沒有好處
   （350 毫秒在筆電上與 250 毫秒同分），而且愈晚補、窗格愈可能把動作開頭擠掉。 */
const RELEASE_HOLD_MS = 250;
/* 手放下之後還來得及補判的時間上限。這個值只在「手放下的那一刻剛好有別的推論在飛」
   時才咬得到（要等它回來才輪到補判）。
   原本這裡寫「真正的期限是 700 − 推論時間，推論超過約 450 毫秒的裝置補判會作廢，
   那是回到現在的行為，不會更糟」。前半句已經由 RELEASE_EXPIRY_HOLD_MS 解掉；
   後半句是錯的——實測每秒 8 格、推論 550 毫秒時，補判 100% 作廢，而且整個網站
   只剩 1.6% 認得出來（改前的舊版也一樣低），並不是「回到原本」那麼無害。
   650 保留不動：推論慢到 650 毫秒還沒回來的裝置，補判會整個不送出（見下面的
   RELEASE_EXPIRY_HOLD_MS 註解末段），那才是這個值真正的邊界。 */
const RELEASE_WINDOW_MS = 650;
/* 補判還在飛的時候，窗格最多可以「不過期」多久。
   窗格在手離開滿 resetAfterMs（700 毫秒）就會被清掉並換世代，補判回來時
   就整個作廢——而「等 250 毫秒 ＋ 推論時間」在手機上（推論約 400 毫秒）
   必然超過 700，所以現行的補判在慢裝置上是<一定>作廢的，不是偶爾。
   這裡讓 continuous-core 在補判飛行期間先不要過期，回來立刻解除。
   上限用 1600：predictLatest 自己有「回來時距送出超過 1500 毫秒就 reset」
   的保險，1600 只是比它多一點，確保萬一結果永遠不回來也不會卡住。 */
const RELEASE_EXPIRY_HOLD_MS = 1600;

const put = (id, text) => { const el = $(id); if (el.textContent !== text) el.textContent = text; };
const store = (k, v) => { try { localStorage.setItem(k, v); } catch {} };
const load = k => { try { return localStorage.getItem(k); } catch { return null; } };

/* 畫面左上角的狀態標籤。示範影片在播的時候顯示「示範」——
   那時候畫面上看到的不是鏡頭，寫「未開啟」或「已暫停」會讓人以為壞了。
   鏡頭本身的狀態仍然記著，切回鏡頭就還原。 */
const STAGE_KIND = {'未開啟': 'off', '啟動中': 'starting', '辨識中': 'live', '已暫停': 'paused', '示範': 'guide'};
let cameraState = '未開啟';
function setStageState(text) {
  if (text) cameraState = text;
  const shown = $('guide-layer').hidden ? cameraState : '示範';
  put('stage-state', shown);
  $('stage-state').dataset.state = STAGE_KIND[shown] || 'off';
}

// 收窄卡片時沿用原本欄位的位置基準，保持右邊界與鏡頭位置不變。
const topbar = document.querySelector('.topbar');
const topbarActions = document.querySelector('.topbar-actions');
const side = document.querySelector('.side');
const viewport = document.querySelector('.viewport');
let sidebarShift = 0, frameShift = 0, settingsShift = 0, brandShift = 0;
const brand = document.querySelector('.brand');
// 手機橫拿（高度不到 500）也不做電腦版的左右對齊：那裡的版面由 style.css 最後的「手機橫拿」一段排好、不用這些位移。
// 寬 901 以上的手機（例如 932×430）原本會落進電腦版的對齊計算——位移沒被用到，卻每次重算都往上累加。
const singleColumn = matchMedia('(max-width: 900px), (max-width: 1100px) and (orientation: portrait), (orientation: landscape) and (max-height: 500px)');
/* 手機、平板直拿時，鏡頭框拿「剩下的高度」。但畫面是 4:3（還沒開鏡頭、教學影片）時，
   框會先被寬度卡住，高度用不完，上下各空一截、框浮在中間。
   多出來的高度（stageSlack）改交給句子框：style.css 用 --stage-slack 把句子框撐高，
   鏡頭那一列就剛好等於框的高度。算法用「加回上次給出去的量」當基準，所以不會來回震盪。 */
const phonePortrait = matchMedia('(max-width: 1100px) and (orientation: portrait)');
const frameBox = document.querySelector('.frame-box');
let stageSlack = 0;
function syncStageSlack() {
  let next = 0;
  if (phonePortrait.matches && frameBox) {
    const box = frameBox.getBoundingClientRect();
    const ar = parseFloat(viewport.style.getPropertyValue('--cam-ar')) || 4 / 3;
    const full = box.height + stageSlack;
    next = Math.max(0, Math.floor(full - Math.min(full, box.width / ar)));
  }
  if (next !== stageSlack) {
    stageSlack = next;
    document.documentElement.style.setProperty('--stage-slack', `${next}px`);
  }
}
const syncPageSpacing = () => {
  document.documentElement.style.setProperty('--topbar-height', `${topbar.getBoundingClientRect().height}px`);
  // 單欄版面（手機、平板直放）不需要左右對齊。條件跟 style.css 的單欄媒體查詢一致。
  if (singleColumn.matches) { sidebarShift = 0; frameShift = 0; settingsShift = 0; brandShift = 0; }
  else {
    const column = side.getBoundingClientRect();
    const frame = viewport.getBoundingClientRect();
    document.documentElement.style.setProperty('--viewport-height', `${frame.height}px`);
    // 圓鈕留在頂列、跟左邊標題同一排（不再往下推到鏡頭框頂端）。
    // 原本推下去是為了跟框的頂端對齊，結果標題在上、圓鈕在下，頂列一高一低。
    settingsShift = 0;
    const tracks = getComputedStyle(document.querySelector('.layout')).gridTemplateColumns;
    const trackWidth = parseFloat(tracks.split(' ').at(-1)) || column.width;
    const available = window.innerWidth - frame.width - trackWidth;
    const middleGap = Math.min(28, Math.max(14, available - 36));
    const gap = (available - middleGap) / 2;
    frameShift = Math.round((gap - frame.left + frameShift) * 1000) / 1000;
    // 右欄四周有一圈留給陰影的內距（style.css 的 --shadow-room-x），對齊的是卡片的右緣，不是右欄的外框。
    const roomRight = parseFloat(getComputedStyle(side).paddingRight) || 0;
    sidebarShift = Math.round((column.right - roomRight + sidebarShift - (window.innerWidth - gap)) * 1000) / 1000;
    // 左上角的標題對齊鏡頭框的左緣（框被移到 gap 的位置）。右上角的圓鈕已經對齊右欄卡片的右緣，
    // 兩邊都對齊，頂列跟下面的內容才在同一組直線上。
    brandShift = Math.round((gap - brand.getBoundingClientRect().left + brandShift) * 1000) / 1000;
  }
  document.documentElement.style.setProperty('--sidebar-shift', `${sidebarShift}px`);
  document.documentElement.style.setProperty('--frame-shift', `${frameShift}px`);
  document.documentElement.style.setProperty('--settings-shift', `${settingsShift}px`);
  document.documentElement.style.setProperty('--brand-shift', `${brandShift}px`);
  syncStageSlack();
};
syncPageSpacing();
const pageSpacingObserver = new ResizeObserver(syncPageSpacing);
[topbar, viewport, side].forEach(element => pageSpacingObserver.observe(element));
window.addEventListener('resize', syncPageSpacing);

/* 畫面上的說明文字，與「要被螢幕報讀唸出來的事件」是兩件事。
   #status 原本掛 role="status"（等於 aria-live），而 onResults 每秒跑約 20 次、
   每一幀都呼叫這裡——手在畫面邊緣時「有看到／沒看到」會一幀一換，
   報讀器就被「等待雙手入鏡／已看到手部／持續辨識中」洗版，
   真正重要的「已辨識：畫畫」排在同一個佇列後面被壓掉。
   現在 #status 只在鏡頭畫面內顯示錯誤，一般狀態不顯示；報讀走 #sr-announce，
   而且預設「會報讀」——每幀那幾個呼叫點才明確傳 announce=false。
   這樣以後新增的訊息預設是安全的（漏報比洗版好救）。 */
/* 鏡頭框中央那一行：平常是入鏡提示；啟動中改成下載進度；鏡頭被關掉時說明原因。
   不另外加一行字——同一個位置換內容，框裡永遠只有一句話。 */
/* 09-25 加「鏡頭跟臉一樣高」：作者提醒手機、平板常常放在桌上往上拍（從下往上的角度），
   訓練影片都是鏡頭在正前方拍的。這句在按「開始辨識」之前出現，正好是擺放裝置的時候。 */
const CAMERA_HINT = '鏡頭跟臉一樣高，上半身和雙手\u200b都入鏡';   // \u200b：最窄的手機（320 寬）折在這裡，不要只剩一個「鏡」
function setCameraHint(text) { put('camera-hint', text || CAMERA_HINT); }

/* 螢幕報讀：先清空再寫，同一句話連續出現（例如連續兩次「已辨識：我」）也一定會被唸。 */
function announce(text) {
  const el = $('sr-announce');
  el.textContent = '';
  requestAnimationFrame(() => { el.textContent = text; });
}

function status(text, error = false, announce = true) {
  put('status', text);
  $('status').classList.toggle('is-error', error);
  $('status').hidden = !error || !text;
  if (announce) put('sr-announce', text);
}

/* ── 教材示範 ────────────────────────────────────────────────
   只播本人錄製、放在 videos/教學/ 的影片；沒有影片的詞顯示「這個詞還沒有示範影片。」
   哪些詞有本人影片由 guide-local.json 決定，這樣不會對不存在的檔案發請求。 */
/* 循環播放時在每一輪之間停半秒。
   無縫循環的問題是看不出動作從哪裡開始、到哪裡結束——
   手語尤其明顯，因為結尾的收手和開頭的舉手會連成一串看不懂的動作。
   停頓讓每一輪有明確的起點。
   （用 loop 屬性加上這個處理，而不是 GIF：同樣清晰度的 GIF 檔案大 10 到 50 倍、
   只有 256 色、而且不吃硬體解碼，手機上會很耗電。） */
const LOOP_PAUSE_MS = 500;
let loopTimer = 0;
function loopWithPause(video) {
  if (video.dataset.looped === '1') return;
  video.dataset.looped = '1';
  video.loop = false;                 // 自己控制，才插得進停頓
  // 看的是外層 #guide-layer：影片本身從來不會被設成 hidden，原本檢查 video.hidden 等於沒檢查，
  // 回到鏡頭時剛好播完的那一輪會在看不見的地方一直循環解碼。
  video.addEventListener('ended', () => {
    if ($('guide-layer').hidden) return;
    clearTimeout(loopTimer);
    loopTimer = setTimeout(() => {
      if ($('guide-layer').hidden) return;
      video.currentTime = 0;
      video.play().catch(() => {});   // 瀏覽器擋自動播放時靜靜略過
    }, LOOP_PAUSE_MS);
  });
}


/* ── 看示範 / 我要比比看 ───────────────────────────
   教學影片與鏡頭共用同一塊大畫面。側欄只有 208px 寬，
   影片塞進去看不清楚手形——那正是學手語唯一要看清楚的東西。 */
let guideSkel = null;          // 目前這個詞的骨架資料
let showSkeleton = false;

/* 切到示範分頁時，辨識必須一起停。原本沒停，造成三件事：
   一、示範層是不透明的，鏡頭畫面完全看不到，但 MediaPipe 仍在全速運算，
       手機會發燙掉電；
   二、點分頁那隻手會經過鏡頭，而窗格不會被清掉，於是使用者「沒有比的詞」
       被記進句子，回頭才發現要一個一個刪；
   三、辨識卡片的 z-index 比示範層高，會浮在示範影片上，還擋住示範自己的
       兩顆按鈕（骨架、0.5 倍速）。
   guideSuspended 記住「是示範分頁把它停掉的」，切回來才知道要不要恢復——
   使用者自己按的暫停不能被覆蓋掉。 */
let guideSuspended = false;

/* 框的長寬比 = 現在看得到的那一層的長寬比。
   鏡頭是 4:3（iOS 直式 3:4），教學影片是 16:9；兩層共用同一個框，
   切分頁時框要跟著換比例，否則其中一邊一定留黑邊。
   還沒開鏡頭、影片也還沒載入時，用 4:3。
   寬度怎麼跟著比例走在 style.css 的 .viewport（桌機與手機做法不同）。 */
function applyFrameRatio() {
  const vp = document.querySelector('.viewport');
  if (!vp) return;
  // 框的形狀只跟著<鏡頭>走，不跟著示範影片變——切換示範／鏡頭時框不動，
  // 示範影片在框裡置中（object-fit: contain），骨架畫布也會算出實際位置對齊。
  // 還沒開鏡頭時用這台裝置最可能的鏡頭形狀：手機、平板直拿是直的 3:4，其他是 4:3，
  // 這樣按下「開始辨識」時框也不會變形。
  const cam = active && video.videoWidth && video.videoHeight;
  const portraitGuess = phonePortrait.matches;
  const w = cam ? video.videoWidth : (portraitGuess ? 3 : 4);
  const h = cam ? video.videoHeight : (portraitGuess ? 4 : 3);
  vp.style.aspectRatio = `${w} / ${h}`;
  vp.style.setProperty('--cam-ar', (w / h).toFixed(4));
  if (typeof syncStageSlack === 'function') syncStageSlack();
}
$('guide-video').addEventListener('loadedmetadata', applyFrameRatio);
// 斷線時示範影片載不到：不要只給一片黑，講出原因
$('guide-video').addEventListener('error', () => {
  put('guide-empty', navigator.onLine === false ? '目前沒有網路，示範影片載不到。' : '示範影片載入失敗，請稍後再試。');
  $('guide-empty').hidden = false;
  $('guide-skeleton').hidden = true;
  $('guide-slow').hidden = true;
});
// 手機轉向（直↔橫）時，還沒開鏡頭的預設形狀也要跟著換
phonePortrait.addEventListener('change', applyFrameRatio);
// 一載入就套用預設形狀（手機直拿 3:4），否則第一次切換畫面時框才變形，整個版面跳一下
applyFrameRatio();

/* 換了影片來源之後馬上 play() 會被載入打斷——實測選詞切到示範時影片是停著的。
   所以先試一次，沒成功就等 loadeddata 再試一次（只掛一次，不會累積監聽器）。 */
function playGuide() {
  const v = $('guide-video');
  if (!v.src) return;
  const go = () => v.play().catch(() => {});
  go();
  if (v.readyState < 2) v.addEventListener('loadeddata', go, { once: true });
}

function setTab(which) {
  const live = which === 'live';
  $('guide-layer').hidden = live;
  applyFrameRatio();
  syncPrimary();
  const v = $('guide-video');
  if (live) {
    clearTimeout(loopTimer);
    try { v.pause(); } catch {}
    // 回到鏡頭時選單回到「25 個詞・選一個看示範」。選單顯示的是「畫面上正在示範哪個詞」，
    // 沒在示範就不該停在某個詞上——而且停著的話，想再看同一個詞會選不動。
    const sel = $('guide-word');
    if (sel && sel.options.length && sel.options[0].value === '') sel.selectedIndex = 0;
    markGuideOption();
  }
  if (typeof syncGuideAgain === 'function') syncGuideAgain();
  else if (v.src) playGuide();

  if (!live) {
    if (active && !paused) { guideSuspended = true; setPaused(true, '看示範時先暫停辨識，按「回到鏡頭」就會繼續。'); }
    $('live-badge').hidden = true;
  } else {
    if (active) $('live-badge').hidden = false;
    if (guideSuspended) { guideSuspended = false; setPaused(false); }
  }
  setStageState();   // 放最後：上面的 setPaused 會寫入鏡頭狀態，這裡再依畫面決定顯示哪個
  // 鏡頭的錯誤訊息（例如「尚未取得鏡頭權限」）是講鏡頭的，播示範時不要蓋在影片上；回到鏡頭再出現。
  // 只有錯誤才顯示——一般訊息是給報讀器的，原本這行會讓它們在畫面上閃一下。
  const st = $('status');
  st.hidden = !live || !st.classList.contains('is-error') || !st.textContent;
}
/* 右欄那一顆主按鈕，一顆做三件事——原本是三顆（開鏡頭、暫停、切分頁）散在兩個地方。
   規則只有一條：<它永遠讓你更靠近「正在比手語」>，標籤寫的就是按下去會發生什麼。
     還沒開鏡頭        → 「我要比比看」開鏡頭
     正在看示範        → 「我要比比看」切回鏡頭（示範時自動暫停的辨識也一起恢復）
     鏡頭開著、辨識中  → 「暫停」
     已暫停            → 「繼續」
   底色代表「畫面現在是鏡頭」，跟標籤各自獨立：暫停時底色仍是深的（畫面還是鏡頭），
   只是標籤變成「繼續」。 */
function syncPrimary() {
  const live = $('guide-layer').hidden;
  const el = $('start-camera');
  el.classList.toggle('is-on', live && active);
  if (fatalRuntime) return;                   // 失敗時的「重新載入」不要被蓋掉
  // 一顆主按鈕講完整個流程：沒開 → 開始辨識；在看示範 → 回到鏡頭；辨識中 → 停止。
  // 「停止」＝關掉鏡頭（句子留著）。不再有「暫停但鏡頭還開著」這個中間狀態——
  // 它需要使用者分辨「暫停」與「關閉」的差別，但再按一下就重開，差別不值得一顆按鈕。
  const label = starting ? '啟動中…' : !active ? '開始辨識' : (!live ? '回到鏡頭' : '停止');
  // 剛變成「停止」的 0.6 秒內不接受停止：在示範畫面連點兩下「回到鏡頭」，第二下不該把鏡頭關掉
  if (label === '停止' && el.textContent !== '停止') stopLockUntil = performance.now() + 600;
  put('start-camera', label);
}

$('start-camera').addEventListener('click', () => {
  if (starting) return;
  /* 辨識元件的主程式（vendor/holistic/holistic.js）本身沒載進來時，重試沒有用——
     它是 <script defer>，不會再跑一次，window.Holistic 永遠是 undefined，按幾次都救不回來
     （2026-09-25 上線後檢查抓到：按鈕承諾了做不到的事）。這種情況直接重新整理頁面。 */
  // 辨識元件下載失敗過（vendorFailed）也一樣：MediaPipe 可能停在一半，重新整理最保險（09-26）
  if (fatalRuntime && (!window.Holistic || vendorFailed)) { location.reload(); return; }
  // 剛按過「停止」的 0.6 秒內不接受「開始辨識」：雙擊「停止」時，第二下會落在已經換好字的按鈕上，
  // 原本會把鏡頭又開起來（2026-09-25 上線後檢查）。
  if (!active && performance.now() < startLockUntil) return;
  if (!active) { setTab('live'); startCamera(); return; }
  if (!$('guide-layer').hidden) { setTab('live'); return; }   // 從示範切回鏡頭
  if (performance.now() < stopLockUntil) return;              // 連點的第二下
  stopCamera();                                               // 辨識中 → 停止
  startLockUntil = performance.now() + 600;                   // 反方向也要鎖（見上面）
});


$('guide-slow').addEventListener('click', () => {
  const b = $('guide-slow'), on = b.getAttribute('aria-pressed') !== 'true';
  b.setAttribute('aria-pressed', String(on));
  $('guide-video').playbackRate = on ? 0.5 : 1;
});
$('guide-skeleton').addEventListener('click', () => {
  const b = $('guide-skeleton');
  showSkeleton = b.getAttribute('aria-pressed') !== 'true';
  b.setAttribute('aria-pressed', String(showSkeleton));
});

/* 骨架疊層。
   骨架是展示用的——它顯示模型看到什麼，但學手形時真實的手比骨架清楚，
   所以預設關閉。
   原本還有一條一直開著的「食指尖軌跡」白線，09-26 拿掉：示範是影片，動作本來就看得到
   （還有 0.5 倍速）；它只追食指尖，雙手詞或整隻手移動時畫出來的線反而誤導；
   臉附近的詞（刷牙、早安）還會蓋住嘴型與手形；而且關不掉。 */
function drawGuideFrame() {
  const v = $('guide-video'), c = $('guide-canvas');
  if (!guideSkel || $('guide-layer').hidden || !v.videoWidth) {
    requestAnimationFrame(drawGuideFrame);
    return;
  }
  const box = v.getBoundingClientRect();
  if (c.width !== Math.round(box.width) || c.height !== Math.round(box.height)) {
    c.width = Math.round(box.width); c.height = Math.round(box.height);
  }
  const g = c.getContext('2d');
  g.clearRect(0, 0, c.width, c.height);

  // 影片用 object-fit: contain，畫布要算出實際顯示的矩形，否則骨架會對不準
  const vr = v.videoWidth / v.videoHeight, cr = c.width / c.height;
  const dw = vr > cr ? c.width : c.height * vr;
  const dh = vr > cr ? c.width / vr : c.height;
  const ox = (c.width - dw) / 2, oy = (c.height - dh) / 2;
  const X = u => ox + u / 1000 * dw, Y = u => oy + u / 1000 * dh;

  const fps = guideSkel.fps || 30;
  const i = Math.min(guideSkel.frames.length - 1, Math.floor(v.currentTime * fps));

  if (showSkeleton) {
    const f = guideSkel.frames[i] || {};
    g.lineCap = 'round';
    if (f.p) {                                   // 身體：細、暗，只是參考框架
      g.strokeStyle = 'rgba(255,255,255,.34)'; g.lineWidth = 2;
      for (const [a, b] of guideSkel.poseEdges || []) {
        const ia = (guideSkel.poseKeep || [0, 11, 12, 13, 14, 15, 16]).indexOf(a);
        const ib = (guideSkel.poseKeep || [0, 11, 12, 13, 14, 15, 16]).indexOf(b);
        if (ia < 0 || ib < 0 || !f.p[ia] || !f.p[ib]) continue;
        g.beginPath(); g.moveTo(X(f.p[ia][0]), Y(f.p[ia][1]));
        g.lineTo(X(f.p[ib][0]), Y(f.p[ib][1])); g.stroke();
      }
    }
    g.strokeStyle = 'rgba(255,255,255,.9)'; g.lineWidth = 2.5;
    for (const key of ['l', 'r']) {              // 手：亮、粗，那是內容
      const h = f[key];
      if (!h) continue;
      for (const [a, b] of guideSkel.handEdges || []) {
        if (!h[a] || !h[b]) continue;
        g.beginPath(); g.moveTo(X(h[a][0]), Y(h[a][1]));
        g.lineTo(X(h[b][0]), Y(h[b][1])); g.stroke();
      }
    }
  }
  requestAnimationFrame(drawGuideFrame);
}
requestAnimationFrame(drawGuideFrame);

function updateGuide() {
  const item = guide.find(x => String(x.id) === $('guide-word').value);
  if (!item) return;
  const localFile = localGuide?.words?.[item.label];
  // 影片內容的版本（產生 guide-local.json 時算的雜湊；公開版未附示範影片）：部署時若讓瀏覽器長時間快取 /videos/*
  // （公開版的 _headers 沒有這條），重錄之後網址要跟著變，不然看過的人會播舊影片、疊新骨架（09-25 重錄六支時發現）
  const ver = localGuide?.['版本']?.[item.label];
  const q = ver ? `?v=${ver}` : '';
  const v = $('guide-video');

  guideSkel = null;
  if (localFile) {
    v.src = `videos/教學/${localFile}${q}`;
    v.playbackRate = $('guide-slow').getAttribute('aria-pressed') === 'true' ? 0.5 : 1;
    $('guide-empty').hidden = true;
    put('guide-empty', '這個詞還沒有示範影片。');
    $('guide-skeleton').hidden = false;
    $('guide-slow').hidden = false;
    loopWithPause(v);
    if (!$('guide-layer').hidden) v.play().catch(() => {});
    // 骨架資料是選配的：沒產生過就只播影片，不要因此壞掉
    // 快速換詞時，上一個詞的骨架可能比較晚回來：只收「現在選的這個詞」的
    const want = item.label;
    const current = () => guide.find(x => String(x.id) === String(lastGuideId ?? $('guide-word').value))?.label;
    fetch(`videos/教學/骨架/${item.label}.json${q}`)
      .then(r => (r.ok ? r.json() : null))
      .then(d => { if (current() === want) guideSkel = d; })
      .catch(() => { if (current() === want) guideSkel = null; });
  } else {
    v.removeAttribute('src');
    $('guide-empty').hidden = false;
    $('guide-skeleton').hidden = true;
    $('guide-slow').hidden = true;
  }
  speech.warm([item.label]);
}

const dataReady = Promise.all([
  fetch('guide.json').then(r => { if (!r.ok) throw Error(); return r.json(); }),
  fetch('model/labels.json').then(r => { if (!r.ok) throw Error(); return r.json(); }),
  fetch('guide-local.json').then(r => r.ok ? r.json() : null).catch(() => null),
]).then(([items, names, local]) => {
  guide = items; labels = names; localGuide = local;
  // 2026-09-29：「同學」要先看到大拇指與小指伸直的手形才能在比到一半時記下（見 continuous-core.js）。
  { const g = Object.keys(names).find(k => names[k] === '同學'); continuous.setHandshapeGate(g === undefined ? null : Number(g)); }

  /* 選詞時自動切到示範畫面，只在<有本人錄製的影片>時才做。
     還沒錄任何一支的時候，切過去只會看到「這個詞還沒有示範影片。」的空畫面，
     等於白按一次。
     放好影片並產生 guide-local.json 之後，
     guide-local.json 裡會有詞，選詞就會直接播示範，不用改程式。 */
  const localCount = Object.keys(localGuide?.words || {}).length;
  if (!localCount) {
    // 一支本人錄的示範都沒有的時候，選詞不該跳到一個空畫面
    guideAuto = false;
    setTab('live');
  }

  fillGuideSelect();
  updateGuide();

  // 從詞彙頁的「用鏡頭試比「刷牙」」過來（網址帶 ?w=刷牙）：
  // 記住這個詞，「再看一次「刷牙」」就在旁邊，開鏡頭比完想對照可以一按回去。
  // 網址上的參數用完就拿掉，重新整理或分享時網址是乾淨的首頁。
  const fromPage = new URLSearchParams(location.search).get('w');
  const picked = fromPage && guide.find(x => x.label === fromPage);
  if (picked) {
    lastGuideId = String(picked.id);
    syncGuideAgain();
  }
  if (fromPage) history.replaceState(null, '', location.pathname);
}).catch(() => {
  status('詞彙資料載入失敗，請重新整理。', true);
  throw Error('Unable to load labels');
});
dataReady.catch(() => {});
/* 選一個詞 = 我想看這個詞怎麼比。直接切到示範，不用再按一顆「看示範」。
   選單要伸手去點，不可能在比手語的時候誤觸，所以這個推論是安全的。 */
let guideAuto = true;
let lastGuideId = null;      // 剛剛看過的詞，給「再看一次」用
function showGuide(id) {
  $('guide-word').value = id;
  markGuideOption();
  lastGuideId = id;
  updateGuide();
  if (guideAuto) { setTab('guide'); playGuide(); }
}
$('guide-word').addEventListener('change', () => showGuide($('guide-word').value));
$('guide-again').addEventListener('click', () => {
  if (!lastGuideId) return;
  showGuide(lastGuideId);
  announce(`示範：${$('guide-word').options[$('guide-word').selectedIndex]?.dataset.word || ''}`);
  $('start-camera').focus();          // 這顆按鈕在示範畫面會藏起來，焦點不要掉到頁面最外層
});

/* 「再看一次『好』」只在「畫面是鏡頭、而且剛剛有看過某個詞」時出現。 */
/* 「再看一次」永遠在同一個位置：看過示範之後就能按（鏡頭畫面時＝切回示範，
   示範畫面時＝從頭再播一次），還沒看過時是灰的。不再忽隱忽現。 */
function syncGuideAgain() {
  const btn = $('guide-again');
  const item = lastGuideId && guide.find(x => String(x.id) === lastGuideId);
  const live = $('guide-layer').hidden;
  btn.hidden = !(live && item && guideAuto);
  // 手機一排三顆時，選單縮成一格、寫著剛剛看的詞（回到鏡頭時選單的值會清空，好讓同一個詞能再選一次，
  // 所以不能看選單的值，要看 lastGuideId）。點它換詞，旁邊的「再看」就是再看這個詞（09-25 驗收：原本寫「換詞」，
  // 回到鏡頭後畫面上看不出剛剛看的是哪個詞）。
  put('guide-word-short', item ? item.label : '換詞');
  if (item) {
    // 詞放在自己的小段落：手機一排三顆時選單已經寫著這個詞，按鈕只留「再看」（style.css 藏 .again-word）
    const word = document.createElement('span');
    word.className = 'again-word';
    word.textContent = `「${item.label}」`;
    const label = document.createElement('span');     // 包成一段：按鈕是 inline-flex＋gap，兩段會被撐開 6 像素
    label.append('再看', word);
    btn.replaceChildren(label);
    btn.setAttribute('aria-label', `再看「${item.label}」`);
  }
  // 手機上只有「鏡頭開著」時才讓它取代選單（style.css 看這個 class）；
  // 鏡頭沒開的時候看到的是選單，第一次來或從詞彙頁進來的人才找得到。
  btn.closest('.guide')?.classList.toggle('has-again', !btn.hidden && active);
}

/* 選單依教材單元分組。
   這裡刻意<不>依「穩定／普通／容易錯」分組（2026-09-24 審查的結論）：
   那個分級是單一種子、單一位考生的一次抽樣，換個種子 25 個詞裡有 7 個會換級；
   每個詞只有 13～25 支考題，做過多重比較校正之後沒有任何一個詞撐得住「穩定」。
   掛在公開網站上等於給使用者一個抽籤抽出來的標籤。
   同一個理由，首頁的準確率卡片也早就拿掉了。 */
function fillGuideSelect() {
  const sel = $('guide-word');
  if (!guide.length) return;
  // 不預選任何詞。下拉選單只有「值改變」時才會通知程式——
  // 原本預設停在「起床」，使用者再選一次「起床」數值沒變，示範就不會播。
  const keep = sel.value || '';
  // \u200b：窄的手機放不下一行時折在「・」後面（09-26）
  sel.replaceChildren(new Option('25 個詞・\u200b選一個看示範', '', !keep, !keep));
  sel.options[0].disabled = true;
  for (const u of [1, 2, 3]) {
    const entries = guide.filter(x => x.unit === u).sort((a, b) => a.seconds - b.seconds);
    if (!entries.length) continue;
    const group = document.createElement('optgroup');
    group.label = `第 ${u} 單元・${entries[0].unitTitle}`;
    entries.forEach(item => {
      const opt = new Option(item.label, String(item.id));
      opt.dataset.word = item.label;
      group.append(opt);
    });
    sel.append(group);
  }
  sel.value = keep;
  if (!sel.value) sel.selectedIndex = 0;          // 回到「25 個詞・選一個看示範」
  markGuideOption();
}

/* 選單顯示的是「被選中那個選項的文字」，所以選中的那個改寫成「示範：多」，
   其他選項維持只有詞本身（下拉清單才不會 25 行都是「示範：」）。 */
function markGuideOption() {
  const sel = $('guide-word');
  for (const opt of sel.options) {
    if (!opt.dataset.word) continue;
    const text = opt.value === sel.value ? `示範：${opt.dataset.word}` : opt.dataset.word;
    if (opt.text !== text) opt.text = text;
  }
  // 疊在選單上、置中顯示的那層字（選單自己的字是透明的）
  put('guide-word-text', sel.options[sel.selectedIndex]?.text || '25 個詞・\u200b選一個看示範');
}

/* ── 主題 ─────────────────────────────────────────────────
   預設跟隨系統設定（這頁有不少必須讀的說明文字）。
   使用者按過切換鈕之後才改用手動設定，鏡頭區則永遠維持深色。 */
const systemDark = window.matchMedia('(prefers-color-scheme: dark)');
function currentTheme() {
  return document.documentElement.dataset.theme
      || (systemDark.matches ? 'dark' : 'light');
}
function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme;
  else delete document.documentElement.dataset.theme;
  const dark = currentTheme() === 'dark';
  $('theme').setAttribute('aria-pressed', String(dark));
  $('theme').title = dark ? '切換為淺色主題' : '切換為深色主題';
}
applyTheme(load('tsl-theme') || null);
$('theme').addEventListener('click', () => {
  const next = currentTheme() === 'dark' ? 'light' : 'dark';
  applyTheme(next);
  store('tsl-theme', next);
});
// 使用者還沒手動選過的話，系統切換主題時跟著走
systemDark.addEventListener('change', () => { if (!load('tsl-theme')) applyTheme(null); });

/* ── 語音 ─────────────────────────────────────────────── */
/* 一顆圓鈕切換：圖示是「現在的聲音」，文字說明寫出按下去會變成什麼。 */
let voiceNow = 'female';
function setVoice(which) {
  voiceNow = which;
  speech.setVoice(which);
  const el = $('voice');
  el.classList.toggle('is-male', which === 'male');
  const label = which === 'male' ? '聲音：男聲（按一下換女聲）' : '聲音：女聲（按一下換男聲）';
  el.setAttribute('aria-label', label);
  el.title = label;
  store('tsl-voice', which);
}
setVoice(load('tsl-voice') === 'male' ? 'male' : 'female');
$('voice').addEventListener('click', () => {
  setVoice(voiceNow === 'male' ? 'female' : 'male');
  speech.say(Object.values(labels).includes('起床') ? '起床' : labels['0'], {force: true});
});

// 自動語音只由新確認的詞觸發；手動播放維持可用。
/* 自動唸出：頂列的一顆喇叭。開著＝辨識到一個詞就唸出來（這是這個網站的重點，
   所以<預設開>）；關掉＝只有按「全部唸出」才出聲。 */
function setAutoSpeech(on) {
  speech.setEnabled(on);
  const el = $('auto-speech');
  el.classList.toggle('is-off', !on);
  el.setAttribute('aria-pressed', String(on));
  el.title = on ? '辨識到的詞會自動唸出（點一下關掉）' : '辨識到的詞不會自動唸出（點一下開啟）';
  store('tsl-auto-speech', String(on));
}
setAutoSpeech(load('tsl-auto-speech') !== 'false');   // 預設開：沒設定過就是開
$('auto-speech').addEventListener('click', () => setAutoSpeech($('auto-speech').classList.contains('is-off')));

/* ── 句子 ─────────────────────────────────────────────── */
/* 正在唸的詞塊亮起來：只切換 class，不整排重建（重建會讓鍵盤焦點掉到頁面最外層）。
   i：-1 沒在唸、-2 整句一起、>=0 第幾個。 */
function markPlaying(i) {
  $('sentence-strip').querySelectorAll('.chip').forEach((chip, k) =>
    chip.classList.toggle('is-playing', i === -2 || k === i));
}

let stripStickToEnd = true;   // 句子框是不是停在最新一排（下面「句子框變大變小」那段用）
function renderSentence() {
  const strip = $('sentence-strip');
  const atEnd = strip.scrollHeight - strip.scrollTop - strip.clientHeight < 4;
  // 重建前記下焦點在第幾個詞、是詞還是 ×，重建完放回同一個位置
  const f = document.activeElement;
  const fChip = f && strip.contains(f) ? f.closest('.chip') : null;
  const fIndex = fChip ? [...strip.querySelectorAll('.chip')].indexOf(fChip) : -1;
  const fKind = f?.classList?.contains('chip-x') ? '.chip-x' : '.chip-word';
  strip.replaceChildren();
  if (!sentence.length) {
    const p = document.createElement('p');
    p.className = 'strip-empty';
    p.textContent = '辨識結果會顯示在這裡';
    strip.append(p);
  } else {
    sentence.forEach((item, i) => {
      const chip = document.createElement('div');
      chip.className = 'chip';
      // 用 <button> 不用 <span>：鍵盤 Tab 得到、報讀器知道可以按。
      const word = document.createElement('button');
      word.type = 'button';
      word.className = 'chip-word';
      word.textContent = item.label;
      word.title = `再聽一次「${item.label}」`;
      word.setAttribute('aria-label', `再聽一次「${item.label}」`);
      word.addEventListener('click', () => {
        markPlaying(i);
        speech.say(item.label, {force: true}).finally(() => markPlaying(-1));
      });
      // 每個詞自己的刪除鈕：辨識錯一個詞是常有的事，不用整句重來
      const x = document.createElement('button');
      x.type = 'button';
      x.className = 'chip-x';
      x.textContent = '\u00d7';
      x.title = `刪掉「${item.label}」`;
      x.setAttribute('aria-label', `刪掉「${item.label}」`);
      x.addEventListener('click', event => {
        event.stopPropagation();
        speech.stop();
        sentence.splice(i, 1);
        renderSentence();                 // 焦點會落到同位置的下一個詞（沒有就上一個）
        if (!sentence.length) $('start-camera').focus();
        announce(`已刪掉「${item.label}」`);
      });
      chip.append(word, x);
      strip.append(chip);
    });
  }
  if (fIndex >= 0) {
    const chips = strip.querySelectorAll('.chip');
    chips[Math.min(fIndex, chips.length - 1)]?.querySelector(fKind)?.focus();
  }
  // 新的詞要看得到；使用者往上翻看舊詞時不要把他拉回來
  if (atEnd) strip.scrollTop = strip.scrollHeight;
  // 有詞才出現這一排；卡片是固定高度，所以只有句子框在卡片裡讓位，卡片不動
  const acts = document.querySelector('.sentence-actions');
  if (acts) acts.hidden = !sentence.length;
  // 唸的時候同一顆按鈕變成「停止」（09-25 作者要求：唸到一半要能停；唸完變回「全部唸出」，再按就是重播）。
  // 不另外加一顆按鈕：兩顆同時出現會被讀成二選一。按鈕寬度由格線固定，換字不會讓卡片動。
  const speakBtn = $('speak-sentence');
  speakBtn.disabled = !sentence.length;
  // 「停止唸出」而不是「停止」：鏡頭開著時主按鈕就叫「停止」（關鏡頭），同一頁兩顆「停止」會按錯（09-25 驗收）
  speakBtn.textContent = speaking ? '停止唸出' : '全部唸出';
  $('clear-sentence').disabled = !sentence.length;
  stripStickToEnd = strip.scrollHeight - strip.scrollTop - strip.clientHeight < 4;
}
renderSentence();

/* 句子框變大變小（轉向：直拿↔橫拿、網址列收合）時：原本看著最新一排，就繼續停在最新一排（09-25 下午）。
   瀏覽器只保留原本的捲動位置（Chrome 轉回直拿甚至歸零），直拿 12 個詞轉成橫拿，看到的是舊的那排。
   「原本看著最新一排」（stripStickToEnd）在兩個時候更新：renderSentence 重建完、使用者自己捲動完。
   框的大小一變，瀏覽器自己調整捲動位置也會觸發 scroll，那一次不是使用者捲的，不算（用框的大小有沒有變來分辨）。
   使用者往上翻看舊詞時不動它。 */
{
  const strip = $('sentence-strip');
  let lastSize = '';
  const sizeKey = () => `${strip.clientWidth}x${strip.clientHeight}`;
  strip.addEventListener('scroll', () => {
    if (sizeKey() !== lastSize) return;
    stripStickToEnd = strip.scrollHeight - strip.scrollTop - strip.clientHeight < 4;
  }, {passive: true});
  new ResizeObserver(() => {
    lastSize = sizeKey();
    if (stripStickToEnd) strip.scrollTop = strip.scrollHeight;
  }).observe(strip);
}

let speakRun = 0;   // 第幾次按「全部唸出」：按停止後馬上再按，舊的那次收尾時不要把新的那次的「停止」蓋掉
let speakStartedAt = 0;   // 剛開始唸的 0.6 秒內不接受「停止唸出」：滑鼠雙擊時第二下會把剛開始的整句停掉（09-26 檢查）
$('speak-sentence').addEventListener('click', async () => {
  if (speaking) {                    // 這時按鈕上寫的是「停止唸出」
    if (performance.now() - speakStartedAt < 600) return;
    speakRun++;
    speech.stop();
    speaking = false; markPlaying(-1); renderSentence();
    announce('已停止。');
    return;
  }
  if (!sentence.length) return;
  const run = ++speakRun;
  speakStartedAt = performance.now();
  speaking = true; renderSentence();
  try { await speech.sayAll(sentence.map(x => x.label), i => { if (run === speakRun) markPlaying(i); }); }
  finally { if (run === speakRun) { speaking = false; markPlaying(-1); renderSentence(); } }
});
$('clear-sentence').addEventListener('click', () => {
  if (!sentence.length) return;
  const count = sentence.length;
  speech.stop();
  sentence.length = 0;
  renderSentence();
  $('start-camera').focus();
  announce(`已清除 ${count} 個詞。`);
});

/* ── 模型與鏡頭 ───────────────────────────────────────── */
function withTimeout(promise, milliseconds, message) {
  let timer;
  const timeout = new Promise((_, reject) => { timer = setTimeout(() => reject(Error(message)), milliseconds); });
  return Promise.race([promise, timeout]).finally(() => clearTimeout(timer));
}

function askWorker(type, values) {
  return new Promise((resolve, reject) => {
    const id = ++requestId;
    const timer = setTimeout(() => { pending.delete(id); reject(Error('辨識元件運算逾時。')); failRuntime('辨識元件運算逾時。'); }, 45000);
    pending.set(id, {resolve, reject, timer});
    try { worker.postMessage({id, type, values}, values ? [values.buffer] : []); }
    catch (error) { clearTimeout(timer); pending.delete(id); reject(error); }
  });
}

/* 第一次開啟要下載約 27 MB（上線網站實測的傳輸量）的手部偵測元件（之後瀏覽器會留著）。

   進度要算「真的正在流進來的位元組」。舊版看 Resource Timing，但那要等一個檔案<整個>下載完
   才會多一筆——17 MB 的 .data 下載期間數字完全不動，慢網路（約 5 Mbps 以下）就會被誤判成斷線，
   其實檔案再過幾秒就下載完了（2026-09-24 全站檢查，用 4 Mbps 實測重現）。
   現在的做法：在載入期間暫時包住 XMLHttpRequest（.data 是用它載的）與 fetch（.wasm、.tflite），
   只處理 /vendor/holistic/ 的請求，一邊把資料原樣交回去、一邊數位元組。不會多發任何請求。
   判定斷線的條件：還有請求沒下載完，而且畫面看得見的時候連續 25 秒沒有任何新位元組。 */
const VENDOR_BYTES = 29.8 * 1000 * 1000;   // .data 16.9 MB＋.wasm 6.5 MB＋.tflite 6.4 MB（解壓後）

function trackVendorDownload(onProgress) {
  const st = { bytes: 0, open: 0, lastAt: performance.now(), failed: false };
  const isVendor = url => String(url || '').includes('/vendor/holistic/');
  const bump = n => { st.bytes += n; st.lastAt = performance.now(); };
  const XHR = window.XMLHttpRequest, openOrig = XHR.prototype.open, fetchOrig = window.fetch;
  XHR.prototype.open = function (method, url, ...rest) {
    if (isVendor(url)) {
      let last = 0;
      st.open++;
      this.addEventListener('progress', e => { bump(e.loaded - last); last = e.loaded; });
      this.addEventListener('loadend', () => { st.open--; st.lastAt = performance.now(); }, { once: true });
      // 下載失敗（連線錯誤、伺服器回錯誤碼）要記下來：原本只看「多久沒有新位元組」，失敗時請求已經結束、
      // open 歸零，永遠判不成中斷，畫面停在「正在啟動…」、按鈕灰掉（09-26 檢查：17 MB 的 .data 單獨失敗時重現）
      this.addEventListener('error', () => { st.failed = true; }, { once: true });
      this.addEventListener('abort', () => { st.failed = true; }, { once: true });
      this.addEventListener('load', () => { if (this.status && ![200, 206, 304].includes(this.status)) st.failed = true; }, { once: true });
    }
    return openOrig.call(this, method, url, ...rest);
  };
  window.fetch = async function (input, init) {
    const url = typeof input === 'string' ? input : input?.url;
    let res;
    try { res = await fetchOrig.call(this, input, init); }
    catch (e) { if (isVendor(url)) st.failed = true; throw e; }
    if (isVendor(url) && !res.ok) st.failed = true;
    if (!isVendor(url) || !res.body) return res;
    st.open++;
    const reader = res.body.getReader();
    let closed = false;
    const done = () => { if (!closed) { closed = true; st.open--; st.lastAt = performance.now(); } };
    const body = new ReadableStream({
      async pull(ctrl) {
        try {
          const { done: end, value } = await reader.read();
          if (end) { done(); ctrl.close(); return; }
          bump(value.byteLength); ctrl.enqueue(value);
        } catch (e) { done(); ctrl.error(e); }
      },
      cancel(reason) { done(); return reader.cancel(reason); },
    });
    // 標頭原樣保留（wasm 的串流編譯要看 Content-Type）
    return new Response(body, { status: res.status, statusText: res.statusText, headers: res.headers });
  };
  let timer = 0;
  const tick = () => {
    // 切到背景時不計時：下載可能被系統暫停，回來不該被判成斷線
    if (document.hidden) st.lastAt = performance.now();
    onProgress(st);
    timer = setTimeout(tick, 500);
  };
  tick();
  return {
    stop() { clearTimeout(timer); XHR.prototype.open = openOrig; window.fetch = fetchOrig; },
    stalled: () => !document.hidden && st.open > 0 && performance.now() - st.lastAt > 25000,
    failed: () => st.failed,
  };
}

function waitWithStallGuard(promise, tracker, message) {
  return new Promise((resolve, reject) => {
    let settled = false;
    promise.then(v => { if (!settled) { settled = true; resolve(v); } },
                 e => { if (!settled) { settled = true; reject(e); } });
    const poll = () => {
      if (settled) return;
      if (tracker.failed()) { settled = true; vendorFailed = true; reject(Error('辨識元件下載失敗，請確認網路後按「重新載入」。')); return; }
      if (tracker.stalled()) { settled = true; reject(Error(message)); return; }
      setTimeout(poll, 1000);
    };
    setTimeout(poll, 1000);
  });
}

async function prepare() {
  if (preparing) return preparing;
  preparing = (async () => {
    await dataReady;
    status('正在載入辨識元件，第一次使用請稍候…');   // 給報讀器；畫面上的進度在框中央
    worker = new Worker(new URL('./inference-worker.js', import.meta.url));
    worker.onmessage = ({data}) => {
      const call = pending.get(data.id); if (!call) return;
      clearTimeout(call.timer); pending.delete(data.id);
      data.type === 'error' ? call.reject(Error(data.message)) : call.resolve(data);
    };
    worker.onerror = () => failRuntime('辨識元件運算中斷。');
    if (!window.Holistic) throw Error('辨識元件載入失敗。');
    holistic = new Holistic({locateFile: file => new URL(`../vendor/holistic/${file}`, import.meta.url).href});
    holistic.setOptions({selfieMode: false, modelComplexity: 1, smoothLandmarks: true,
      enableSegmentation: false, refineFaceLandmarks: false,
      minDetectionConfidence: .5, minTrackingConfidence: .5});
    holistic.onResults(onResults);
    // 進度顯示在鏡頭框中央（原本寫進 #status，但一般訊息不顯示，等於沒有進度）。
    // 下載完之後還要編譯 wasm、初始化，手機上要好幾秒：那段改寫「正在啟動…」，不要凍在某個百分比。
    const tracker = trackVendorDownload(({ bytes, open }) => {
      if (!starting) return;
      if (open > 0 || bytes === 0) {
        const pct = Math.min(99, Math.round(bytes / VENDOR_BYTES * 100));
        setCameraHint(bytes ? `正在下載辨識元件 ${pct}%（只有第一次）` : '正在準備辨識元件…');
      } else {
        setCameraHint('正在啟動…');
      }
    });
    try {
      await Promise.all([
        askWorker('init'),
        waitWithStallGuard(holistic.initialize(), tracker,
          '辨識元件下載中斷，請確認網路。已下載的部分會保留。'),
      ]);
    } finally { tracker.stop(); }
  })().catch(error => { preparing = null; failRuntime(error.message || '辨識元件無法啟動。'); throw error; });
  return preparing;
}

/* 目標人形：畫一個「站在正確位置」的頭與肩膀，讓人把自己對上去。
   （2026-09-25 作者要求：只有文字提示時，使用者不知道要退多少才算夠。
     第一版用虛線畫成頭圈＋梯形，作者說「好醜」，改成一條連續的細線、圓肩、柔和陰影。）
   位置與大小的依據跟 framing-hint.js 同一套：肩線在畫面高 62%、肩寬占畫面寬 1/3.4
   （＝訓練資料 7,167 支片段的中位數 3.49 個肩寬，落在最準的那一段）。
   只畫頭與肩，不畫手——手要能自由移動，畫出來會變成「手要放這裡」的誤導。
   位置不對時亮一點、對了就淡下去；記下第一個詞之後整個不畫（見 showTargetPose）。 */
function drawTargetPose(strong) {
  const W = overlay.width, H = overlay.height;
  const sw = W / 3.4;                       // 目標肩寬
  const sy = H * 0.62;                      // 目標肩線
  const cx = W / 2;
  const hrx = sw * 0.33, hry = sw * 0.40;   // 頭（略長的橢圓，照訓練資料的頭肩比例）
  const hy = sy - sw * 0.74;
  const c = overlayContext;
  c.save();
  c.lineCap = 'round';
  c.lineJoin = 'round';
  c.lineWidth = 2.5;
  // 用陰影當描邊：淺色背景與深色背景上都看得見，又不會像雙線那麼笨重
  c.shadowColor = 'rgba(0, 0, 0, .55)';
  c.shadowBlur = 7;
  c.strokeStyle = strong ? 'rgba(255,255,255,.95)' : 'rgba(255,255,255,.42)';

  c.beginPath();
  c.ellipse(cx, hy, hrx, hry, 0, 0, Math.PI * 2);
  c.stroke();

  // 脖子與肩膀：圓肩，不是梯形的硬角
  const ny = hy + hry * 0.92;               // 脖子接到頭的位置
  const nx = sw * 0.13;                     // 脖子半寬
  const sx = sw / 2;                        // 肩點
  c.beginPath();
  c.moveTo(cx - nx, ny);
  c.lineTo(cx - nx, sy - sw * 0.10);                         // 脖子
  c.quadraticCurveTo(cx - sx * 0.82, sy - sw * 0.02, cx - sx, sy + sw * 0.06);   // 左肩圓角
  c.lineTo(cx - sw * 0.62, H);                               // 左側身體
  c.moveTo(cx + nx, ny);
  c.lineTo(cx + nx, sy - sw * 0.10);
  c.quadraticCurveTo(cx + sx * 0.82, sy - sw * 0.02, cx + sx, sy + sw * 0.06);
  c.lineTo(cx + sw * 0.62, H);
  c.stroke();
  c.restore();
}

/* 姿態的前 11 個點是臉（眼、耳、嘴），畫出來就是在臉上塗一團藍線——畫面最亂的來源，
   而且辨識用不到它們判斷手語。2026-09-25 作者說「畫面不要看起來很雜亂」，只留肩、手臂、髖。 */
const FACE_POSE_MAX = 10;

function draw(landmarks, connections, color, skipTo = -1) {
  if (!landmarks) return;
  overlayContext.strokeStyle = color; overlayContext.fillStyle = color; overlayContext.lineWidth = 2;
  for (const [a, b] of connections) {
    if (a <= skipTo || b <= skipTo) continue;
    const p = landmarks[a], q = landmarks[b];
    if (!p || !q || (p.visibility ?? 1) < .35 || (q.visibility ?? 1) < .35) continue;
    overlayContext.beginPath();
    overlayContext.moveTo(p.x * overlay.width, p.y * overlay.height);
    overlayContext.lineTo(q.x * overlay.width, q.y * overlay.height);
    overlayContext.stroke();
  }
  landmarks.forEach((p, i) => {
    if (i <= skipTo || (p.visibility ?? 1) < .35) return;
    overlayContext.beginPath();
    overlayContext.arc(p.x * overlay.width, p.y * overlay.height, 2.5, 0, 2 * Math.PI);
    overlayContext.fill();
  });
}

/* 準確率卡已移除（2026-09-23）。
   原本這裡有 loadFacts()，從 model-facts.json 讀數字填進卡片。
   拿掉的理由：數字綁模型版本，
   而網站上的數字跟報告對不上是最難解釋的狀況。
   現在改成一句不綁版本的白話說明，沒有任何東西需要在重訓後同步。 */

/* 辨識卡片的文字不能跟著每一幀跳。
   onResults 每秒跑約 20 次，手在畫面邊緣時「有偵測到／沒偵測到」會一幀一換，
   畫面就在「請讓雙手入鏡」「觀察你的動作」「確認中」之間閃個不停——使用者看到的是抖動，不是狀態。
   規則：
     一般狀態要<連續維持 400 毫秒>才真的換上去（同一個目標文字連續出現才算）
     「已辨識」立刻顯示，而且至少停 1200 毫秒，期間其他狀態進不來
   這只影響畫面上的字，辨識邏輯本身不延遲。 */
const badge = {
  shown: null, pending: null, pendingSince: 0, holdUntil: 0,
  SETTLE_MS: 400, HOLD_MS: 1200,
  _apply(t) {
    put('live-word', t.word); put('live-state', t.state); put('live-score', t.score);
    $('live-badge').classList.toggle('is-stable', !!t.stable);
    $('live-badge').classList.toggle('is-idle', !!t.idle);
    if (t.idle) renderAlts([]);
    this.shown = t.key; this.pending = null;
  },
  request(word, state, score, idle) {
    const now = performance.now();
    if (now < this.holdUntil) return;                       // 「已辨識」還在停留
    const t = { word, state, score, idle, key: `${word}|${state}|${idle}` };
    if (t.key === this.shown) { this.pending = null; return; }
    if (this.pending !== t.key) { this.pending = t.key; this.pendingSince = now; return; }
    if (now - this.pendingSince >= this.SETTLE_MS) this._apply(t);
  },
  commit(word, state, score) {
    this._apply({ word, state, score, idle: false, stable: true, key: `${word}|${state}|commit` });
    this.holdUntil = performance.now() + this.HOLD_MS;
  },
  /* 只清內部記錄的話，再開鏡頭後卡片會掛著上一次的字 0.4 秒（2026-09-25 上線後檢查），畫面也一起清。 */
  reset() {
    this.shown = null; this.pending = null; this.holdUntil = 0;
    put('live-word', ''); put('live-state', ''); put('live-score', '');
    $('live-badge').classList.remove('is-stable'); renderAlts([]);
  },
};

function showWaiting(word, state = '觀察中', score = '', idle = false) {
  badge.request(word, state, score, idle);
}

/* 顯示第二、三名。
   模型不確定的時候，正確答案常常就在這裡；只顯示第一名會讓使用者
   以為系統壞掉。信心數值本身不可靠，所以只列詞、不列百分比；
   電腦版用淡淡的長條表示相對高低。
   低於 3% 的候選不列——那已經是雜訊，列出來只會洗版。 */
function renderAlts(ranked) {
  const box = $('live-alts');
  if (!box) return;
  const alts = ranked.slice(1, 3).filter(item => item.score >= 0.03);
  if (!alts.length) { if (box.childElementCount) box.replaceChildren(); return; }
  const top = ranked[0] ? ranked[0].score : 1;
  box.replaceChildren(...alts.map(item => {
    const li = document.createElement('li');
    li.style.setProperty('--bar', `${Math.round(item.score / Math.max(top, 1e-6) * 100)}%`);
    const w = document.createElement('span');
    w.className = 'alt-word'; w.textContent = item.label;
    li.append(w);            // 不顯示百分比：信心數字撐不住（見 CONFIRM_THRESHOLD 的註解）
    return li;
  }));
}

function onResults(results) {
  if (!active || paused || processingGeneration !== generation
      || processingRecognitionGeneration !== continuous.generation) return;
  const now = performance.now();
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  /* 還沒成功記下詞、或位置不對的時候，畫出目標骨架讓人對位；對好、開始認得出詞就不再畫，
     免得一直有東西疊在畫面上（使用者的規則：畫面越簡單越好）。 */
  if (showTargetPose) drawTargetPose(!!framingIssue);
  draw(results.poseLandmarks, POSE_CONNECTIONS, '#5AA9FF', FACE_POSE_MAX);
  draw(results.leftHandLandmarks, HAND_CONNECTIONS, '#FFC178');
  draw(results.rightHandLandmarks, HAND_CONNECTIONS, '#2DD4A7');
  latestFrameValid = validFraming(results);
  const state = continuous.push(extractKeypoints(results), latestFrameValid, now);
  /* 位置提示：只在「手沒在比」（手不在畫面、或手停著）的幀量位置，也只在那時候顯示。
     framing 自己取最近 1 秒的中位數，同一個問題要持續 1 秒才出現、解除 1 秒才收起。 */
  const idle = !latestFrameValid || (!state.ready && state.reason === 'not_moving');
  const issue = framing.update(results.poseLandmarks, inputCanvas.width, inputCanvas.height, now, idle);
  /* 真的擋路的（沒看到人、太近、太遠、胸口被切、偏一邊）隨時可以講；建議（會更準）只在取景階段講。
     例外「太近」（2026-09-25 上線後檢查）：網站示範影片的取景是 2.24～2.71 個肩寬，全部低於太近門檻，
     也就是照那樣的坐法，卡片會整場掛著「請往後坐一點」——即使詞一個一個都認出來了。
     偵測沒有錯（退後確實更準），錯的是「已經在成功辨識了還一直糾正」。所以只要這一輪已經記下詞，
     「太近」就降級成建議：取景階段照講，成功之後閉嘴。其他擋路的（沒看到人、胸口被切…）不適用，
     那些情況下根本記不到詞。 */
  const nagging = issue === 'too_close' && !framingCoaching;
  // 開鏡頭頭 5 秒在講「鏡頭跟臉一樣高」：這段時間只讓擋路的問題（沒看到人、太近…）插話，
  // 「手機橫著拿會更好比」「往後坐一點會更準」這種建議等 5 秒後再講（09-25 檢查：直拿時 1.6 秒就被蓋掉）
  const eyeNow = framingCoaching && performance.now() - cameraStartedAt < EYE_PROMPT_MS;
  const hint = issue && !nagging && (isHard(issue) || (framingCoaching && !eyeNow)) ? FRAMING_TEXT[issue] : '';
  framingIssue = issue && !nagging ? issue : '';
  /* 目標人形什麼時候消失（2026-09-25 作者問「是認到第一個詞就消失還是對準才消失？」）：
     <對準才消失>，不是認到詞就消失。理由：位置不對時認出來的詞比較可能是錯的，
     用「認到一個詞」當作過關，等於在獎勵一個不好的位置。
       · 還沒記下第一個詞：只要位置還有任何問題（含「會更準」這種建議），就繼續畫；
         位置乾淨滿 0.8 秒才淡出（alignedSince），漂掉就立刻再出現。
       · 已經記下詞之後：只有真的會讓辨識做不起來的問題才再出現（沒看到人、胸口被切、偏一邊、太遠）；
         「太近」不再出現——詞都認得出來了還一直畫框是干擾（跟文字提示的規則一致）。
     位置的判斷只在「手沒在比」的幀更新（framing.update 的 idle），所以比到一半不會突然冒出來。 */
  if (issue) alignedSince = 0;
  else if (!alignedSince) alignedSince = now;
  showTargetPose = framingCoaching
    ? (!alignedSince || now - alignedSince < 800)
    : !!(issue && isHard(issue) && !nagging);

  if (!latestFrameValid) {
    /* 原本這裡一律顯示「讓頭、手與上半身入鏡」。但這個分支最常見的情況是：人坐得好好的，
       只是手放著沒比——那是「準備好了」，不是錯誤；而且上一句剛叫使用者「手放下再比一次」，
       這裡馬上說使用者沒入鏡，前後矛盾。現在分三種情況講：
         沒看到人 → 請坐到鏡頭前（framing 給 no_person）
         人在、位置有問題 → 講那個問題
         人在、位置沒問題（手放下） → 開始比一個詞 */
    showWaiting(hint || idlePrompt(), '', '', true);
    // 手離開畫面<不等於>沒東西可判。窗格裡的影格要等 700 毫秒才過期，
    // 而「剛比完、手放下」那一刻的窗格正好是整個動作最完整的一段。
    // 原本這裡直接 return，那個最完整的窗格永遠不會被送進模型——
    // 實測「沒反應」大部分是這樣來的：確認要連續三次、每 250 毫秒一次，
    // 最快也要手出現後 1,250 毫秒才湊得到第三次，而「他」「我」「你」這些
    // 短詞的手可見時間中位數只有 1,200–1,275 毫秒。
    // 所以在這裡補推論一次，帶 released = true 讓 observe 用放下手的規則。
    if (handsGoneSince === null) { handsGoneSince = now; releaseDone = false; }
    const gone = now - handsGoneSince;
    if (gone >= RELEASE_HOLD_MS) lowerHint = false;   // 手離開滿 250 毫秒才算放下；MediaPipe 偶爾漏抓一格手不算
    // 只補一次：手已經離開，窗格不會再變，同一份輸入推論幾次結果都一樣。
    // 而且愈早補愈好——窗格只保留最近 2.5 秒，拖下去長一點的動作開頭會被擠掉。
    // 這裡故意<不>檢查 state.ready：手放下之後窗格的「無效影格比例」一定很高，
    // 那道關卡要交給 makeInput(now, true, true) 用 released 的規則判，不能在這裡先擋掉。
    if (!releaseDone && gone >= RELEASE_HOLD_MS && gone < RELEASE_WINDOW_MS && !inFlightPrediction) {
      releaseDone = true;
      void predictLatest(now, true);
    }
    return;
  }
  handsGoneSince = null; releaseDone = false;
  if (!state.ready) {
    if (state.reason === 'not_moving') {
      // 模型只有 25 類、沒有「沒在比」這一類，靜止的手一定會被分到某個詞。
      // 擋下來之後要講清楚，否則使用者以為當掉了。
      lowerHint = false;                // 手停住不動＝放下了（手放在畫面下緣時偵測得到但不會動）
      showWaiting(hint || idlePrompt(), '', '', true);
    } else if (state.reason === 'insufficient_valid_frames' && state.frameIntervalMs > 220) {
      // 條件放寬後仍湊不滿，代表這台裝置真的太慢。講出原因，
      // 不要讓畫面無限停在「正在觀察動作」而不給任何線索。
      // 只在卡片上講一次（它有 400 毫秒的穩定處理，不會閃）；不再同時在框頂跳錯誤框
      showWaiting('裝置處理速度不足', '', '', true);
    } else {
      showWaiting(lowerHint ? LOWER_PROMPT : '觀察你的動作', '', '', true);
    }
    return;
  }
  if (!inFlightPrediction && now - lastPredictionAt >= INFERENCE_INTERVAL_MS) void predictLatest(now);
}

async function predictLatest(now, released = false) {
  const values = continuous.makeInput(now, true, released);   // true = 加做鏡像版，兩者平均
  if (!values) { if (released) releaseDone = false; return; }
  const cameraGeneration = generation, recognitionGeneration = continuous.generation;
  const serial = ++inferenceSerial; inFlightPrediction = serial; lastPredictionAt = now;
  try {
    // 補判：在它回來之前先別讓窗格過期，否則慢裝置上這一次<一定>白跑（見 RELEASE_EXPIRY_HOLD_MS）。
    // 放在 try 裡面（審查建議）：萬一這一行丟錯，finally 仍會把 inFlightPrediction 歸零，辨識不會整個停住。
    if (released) continuous.holdExpiry(now + RELEASE_EXPIRY_HOLD_MS);
    const result = await askWorker('predict', values);
    // 原本這裡有 `|| !latestFrameValid`：只要「推論回來的那一瞬間」手已經
    // 放下，整個結果就被丟掉。但比完一個詞把手放下是最自然的收尾動作，
    // 等於最後一個詞常常完全不會出現。窗格本身還在（手離開要滿 700 毫秒
    // 才清），所以這裡改成不看單一張最新影格。
    if (!active || paused || cameraGeneration !== generation
        || recognitionGeneration !== continuous.generation) return;
    if (performance.now() - now > 1500) { continuous.reset(); showWaiting(lowerHint ? LOWER_PROMPT : '觀察你的動作', '', '', true); return; }
    const decision = continuous.observe(result.scores, CONFIRM_THRESHOLD,
                                        performance.now(), recognitionGeneration, released);
    if (recognitionGeneration !== continuous.generation || !decision.top) {
      // 手還在動就維持同一句，不要在「手放下…」和「觀察你的動作」之間來回跳（卡片 400 毫秒才換字，來回跳會兩句都換不上去）
      showWaiting(lowerHint ? LOWER_PROMPT : '觀察你的動作', '', '', true); return;
    }
    const ranked = rankScores(result.scores, labels), best = ranked[0];
    const pct = '';                     // 不顯示信心百分比（撐不住的數字）
    if (performance.now() >= badge.holdUntil) renderAlts(ranked);
    // 「已辨識」必須看 confirmed，不能看 stable。兩者不同：stable 只表示
    // 「同一個詞連續穩定出現」，confirmed 才是真的寫進句子。原本看 stable，
    // 於是冷卻期間畫面顯示「已辨識」但句子其實沒有動。
    if (decision.confirmed) {
      badge.commit(best.label, '已辨識', pct);
      lowerHint = true;
    } else if (lowerHint) {
      showWaiting(LOWER_PROMPT, '', '', true);
    } else if (decision.reason === 'awaiting_release') {
      // 同一個詞要再記一次，必須先放開（手放下或停一下）。講清楚，
      // 不然使用者會以為當掉了，或一直重比。
      // 標籤縮短：前一個狀態已經顯示過「已辨識」，這裡重複只會把
      // 最長的詞（「我們兩個」）在 375px 的手機上擠成「我們兩…」。
      showWaiting(best.label, '手放下再比一次', pct);
    } else if (decision.reason === 'cooling_down') {
      showWaiting(best.label, '剛辨識過，稍候', pct);
    } else {
      // 一直認不出來（超過 6 秒沒記錄到新詞）時，把「信心不足」換成下一步該怎麼做
      const stuck = performance.now() - lastConfirmedAt > 6000;
      // 認不出來的原因如果是位置（坐太近、手機直拿、手一直出框），「放慢一點」是錯的建議：先講位置。
      // 這裡是狀態列（小字），不是大字，所以連「建議」等級的也可以講——這時使用者已經卡住了，是會想聽的時候。
      const fix = framing.shown ? FRAMING_TEXT[framing.shown] : '';
      showWaiting(decision.qualified ? best.label : '還不確定',
                  decision.qualified ? '確認中' : (stuck ? (fix || '手放下再比') : '信心不足'), pct);
    }
    if (decision.confirmed) {
      const entry = decision.confirmed;
      const label = labels[String(entry.index)];
      sentence.push({label, score: entry.score});
      if (!speaking) speech.enqueue(label);   // 正在「全部唸出」時不自動唸新詞，不然整句會被切斷（09-26 檢查）；新詞照樣加進句子
      lastConfirmedAt = performance.now();
      framingCoaching = false;      // 已經成功比出一個詞：取景過關了，之後不再主動給「會更準」的建議
      if (sentence.length > SENTENCE_LIMIT) sentence.shift();
      renderSentence();
      announce(`已辨識：${label}`);      // 每次都唸，連續兩次同一個詞也唸
      speech.prefetchSentence?.(sentence.map(x => x.label));   // 先在背景把整句合成好
    }
  } catch (error) {
    if (cameraGeneration === generation && recognitionGeneration === continuous.generation)
      failRuntime(error.message || '辨識元件運算中斷。');
  } finally {
    if (inFlightPrediction === serial) inFlightPrediction = 0;
    // 補判已經判完（確認、信心不足、逾時、世代不符都算），立刻把過期交還給
    // continuous-core：下一張影格就會照原本的規則清掉窗格。
    if (released) continuous.holdExpiry(0);
  }
}

function syncDimensions() {
  const height = Math.round(640 * (video.videoHeight || 480) / (video.videoWidth || 640));
  if (inputCanvas.width === 640 && inputCanvas.height === height) return;
  inputCanvas.width = 640; inputCanvas.height = height;
  overlay.width = 640; overlay.height = height;
  // 鏡頭長寬比校正：MediaPipe 的 y 是除以畫面高、x 是除以畫面寬，而前處理只用肩寬
  // （一個 x 方向的長度）當尺度，所以垂直距離會多乘一個「畫面寬÷高」。
  // 把實際畫布的比例告訴 preprocess.js，它會在送進模型前把 y 換算回 16:9 的尺度。
  // 這是送給 MediaPipe 的那張畫布，所以用 inputCanvas 而不是 video。
  setFrameAspect(inputCanvas.width, inputCanvas.height);
  // 框寫死 4:3，但 iOS 直式回傳的是 3:4，影像會被塞成一條細長的直條、
  // 左右各留一大片黑邊。讓框跟著串流走，黑邊就消失。
  applyFrameRatio();
  continuous.reset(); framing.reset(); latestFrameValid = false;
  handsGoneSince = null; releaseDone = false; lowerHint = false; badge.reset();
}

async function frameLoop(time, token) {
  if (!active || token !== generation) return;
  const next = () => { if (active && token === generation) loopId = requestAnimationFrame(t => frameLoop(t, token)); };
  if (paused || time - lastFrame < 50) { next(); return; }
  lastFrame = time;
  try {
    if (video.readyState >= 2) {
      syncDimensions();
      inputContext.save();
      inputContext.translate(inputCanvas.width, 0); inputContext.scale(-1, 1);
      inputContext.drawImage(video, 0, 0, inputCanvas.width, inputCanvas.height);
      inputContext.restore();
      processingGeneration = token; processingRecognitionGeneration = continuous.generation;
      frameTask = withTimeout(holistic.send({image: inputCanvas}), 10000, '辨識元件沒有回應。');
      await frameTask;
    }
  } catch (error) {
    // 只有「還在跑的這個世代」才算真的壞了。關掉鏡頭或切到背景之後，
    // 上一幀的 10 秒逾時會晚一點才 reject；原本不分世代直接 failRuntime，
    // 使用者回來就看到「辨識元件沒有回應」的致命錯誤，只能重新整理。
    if (active && token === generation) failRuntime(error.message || '辨識元件已中斷。');
    return;
  }
  next();
}

async function startCamera() {
  // 重試不要用 location.reload()——那會把已經下載好的元件丟掉重來，
  // 而且會連同辨識出的句子一起清掉。直接重跑 prepare()，
  // 瀏覽器快取裡下載完成的部分都還在。
  if (fatalRuntime) {
    fatalRuntime = false; preparing = null;
    $('camera-empty').hidden = false;
    syncPrimary();
    status('重新載入中…');
  }
  if (active || starting || $('start-camera').disabled) return;
  const token = ++generation;
  starting = true;
  /* 按鈕 disabled 的那一刻瀏覽器會把焦點拿走、重新啟用時不會還回來：
     用鍵盤的人要再按 7 次 Tab 才回得到主按鈕（2026-09-25 上線後檢查）。先記下來，啟用後還給它。 */
  const hadFocus = document.activeElement === $('start-camera');
  $('start-camera').disabled = true;
  syncPrimary();
  setStageState('啟動中');
  setCameraHint('正在準備辨識元件…');
  try {
    if (!isSecureContext || !navigator.mediaDevices?.getUserMedia)
      throw Error('這個瀏覽器不能開鏡頭，請改用 Chrome、Edge 或 Safari 開啟。');
    await prepare(); if (token !== generation) return;
    // 等上一輪還在飛的那一幀落地就好，但不要讓它的失敗決定這一輪的生死。
    await frameTask.catch(() => {});
    if (token !== generation || fatalRuntime) return;
    setCameraHint('請允許使用鏡頭（不需要麥克風）');
    status('請允許使用鏡頭。允許後就會自動開始辨識，不需要麥克風。');
    const incoming = await navigator.mediaDevices.getUserMedia({audio: false,
      video: {facingMode: 'user', width: {ideal: 640}, height: {ideal: 480}, frameRate: {ideal: 30, max: 30}}});
    if (token !== generation) { incoming.getTracks().forEach(t => t.stop()); return; }
    /* 啟動途中就切到別的分頁：「切走就關鏡頭」只看 active，擋不到啟動中這一段，
       鏡頭會在背景一直開著（2026-09-25 上線後檢查）。拿到鏡頭時頁面已經看不見，就直接收掉。 */
    if (document.hidden) {
      incoming.getTracks().forEach(t => t.stop());
      starting = false; $('start-camera').disabled = false; stopCamera('離開頁面時鏡頭已自動關掉');
      return;
    }
    stream = incoming; video.srcObject = stream;
    await withTimeout(video.play(), 15000, '鏡頭沒有提供畫面，請關閉其他視訊程式後重試。');
    if (token !== generation) { incoming.getTracks().forEach(t => t.stop()); return; }
    continuous.reset(); framing.reset(); framingCoaching = true; alignedSince = 0; latestFrameValid = false;
    handsGoneSince = null; releaseDone = false; lowerHint = false; lastPredictionAt = -Infinity; syncDimensions(); badge.reset();
    lastConfirmedAt = performance.now();   // 剛開鏡頭不算「一直認不出來」
    active = true; paused = false; starting = false;
    $('start-camera').disabled = false;
    if (hadFocus && (document.activeElement === document.body || !document.activeElement)) {
      $('start-camera').focus({preventScroll: true});
    }
    video.hidden = false; overlay.hidden = false;
    $('camera-empty').hidden = true; $('live-badge').hidden = false;
    setCameraHint();
    setStageState('辨識中');
    cameraStartedAt = performance.now();
    showWaiting(EYE_PROMPT, '', '', true);
    status('鏡頭已開啟。鏡頭跟臉一樣高、上半身和雙手入鏡，就可以開始比。');
    speech.warm(Object.values(labels));
    stream.getVideoTracks().forEach(track => track.addEventListener('ended', () => {
      if (active) stopCamera('鏡頭中斷了');
    }));
    lastFrame = 0; loopId = requestAnimationFrame(t => frameLoop(t, token));
    // 啟動期間使用者去選詞看示範了：不要把他從示範拉走，
    // 鏡頭在背景準備好、辨識先暫停，按鈕會寫「回到鏡頭」
    if (!$('guide-layer').hidden) {
      guideSuspended = true; setPaused(true, '');
      $('live-badge').hidden = true;
      syncPrimary(); setStageState(); syncGuideAgain();
    } else {
      setTab('live');
    }
  } catch (error) {
    starting = false;
    if (token !== generation) return;
    stopCamera();
    /* 權限被拒是第一次使用最常見的失敗，修法不能只對桌機成立：
       iPhone Safari 沒有「網址列的網站權限」，而且已經拒絕過之後
       重新整理不會再問一次，叫人重新整理等於導向一個無效動作。 */
    const text = error.name === 'NotAllowedError'
      ? '尚未取得鏡頭權限。iPhone：點網址列左邊的「ᴀA」→ 網站設定 → 相機 → 允許；'
        + '電腦與 Android：點網址列左邊的圖示允許相機。設定完請重新整理。'
      : error.name === 'NotFoundError' ? '找不到鏡頭，請接上攝影機或換一台裝置。'
      : error.name === 'NotReadableError' ? '鏡頭可能正被其他程式使用，請關閉後重試。'
      : error.message;
    status(text || '無法開啟鏡頭，請重新整理，或換一個瀏覽器開啟。', true);
  }
}

/* 暫停／繼續的單一入口。使用者按「暫停」與切到示範分頁走的是同一條路徑，
   兩邊各寫一次的話遲早會漏掉其中一個清理步驟。 */
function setPaused(flag, message = null) {
  if (!active || paused === flag) return;
  paused = flag;
  continuous.reset(); framing.reset(); framingCoaching = true; alignedSince = 0; latestFrameValid = false;
  handsGoneSince = null; releaseDone = false; lowerHint = false; lastPredictionAt = -Infinity; badge.reset();
  lastConfirmedAt = performance.now();   // 剛繼續也一樣
  if (paused) speech.stop();
  syncPrimary();
  setStageState(paused ? '已暫停' : '辨識中');
  overlay.hidden = paused;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  showWaiting(paused ? '已暫停' : '觀察你的動作', '', '', true);
  status(message || (paused ? '已暫停辨識。' : '已繼續辨識，可以直接比手語。'));
}


function stopCamera(message = null) {
  // 注意：這個函式同時當事件處理器用，所以參數可能是一個 Event，不是字串。
  const note = typeof message === 'string' ? message : null;
  const wasActive = active;
  generation++; active = false; paused = false; guideSuspended = false; starting = false;
  // 把還在飛的那一幀的 rejection 吞掉並換成乾淨的 Promise，
  // 否則下次開鏡頭會先卡在舊的逾時上。
  frameTask.catch(() => {}); frameTask = Promise.resolve();
  continuous.reset(); framing.reset(); latestFrameValid = false;
  handsGoneSince = null; releaseDone = false; lowerHint = false; cancelAnimationFrame(loopId); badge.reset();
  speech.stop();
  stream?.getTracks().forEach(track => track.stop());
  stream = null; video.srcObject = null; video.hidden = true; overlay.hidden = true;
  overlayContext.clearRect(0, 0, overlay.width, overlay.height);
  $('camera-empty').hidden = false; $('live-badge').hidden = true;
  $('start-camera').hidden = false; $('start-camera').disabled = false;
  setStageState('未開啟');
  syncPrimary();
  syncGuideAgain();
  // 被動關掉（離開頁面、鏡頭中斷）時，框中央那一行說明原因；平常關掉就回到入鏡提示。
  setCameraHint(note);
  // 下面這些只給報讀器（畫面上不顯示一般訊息）
  status(note || (wasActive
    ? (sentence.length ? '鏡頭已關閉。句子還留著，可以再播放或清除。' : '鏡頭已關閉。')
    : '已取消啟動。'));
}

/* 只讓自家寫好的中文訊息出現在畫面上。原本直接把 error.message 貼出去，
   於是載入失敗時使用者會看到 "Unable to load labels" 之類的內部字串，
   而且會蓋掉原本那句正確的中文提示。 */
function friendlyMessage(message) {
  if (typeof message === 'string' && /[\u4e00-\u9fff]/.test(message)) return message;
  // 英文的內部錯誤不一定是網路造成的，不要一律怪網路；真的離線才講網路
  return navigator.onLine === false ? '目前沒有網路，辨識元件無法下載。' : '辨識元件停止運作了，請按「重新載入」。';
}

function failRuntime(message) {
  if (fatalRuntime) return;
  if (message && !/[\u4e00-\u9fff]/.test(String(message))) console.error('[failRuntime]', message);
  fatalRuntime = true; stopCamera();
  for (const call of pending.values()) { clearTimeout(call.timer); call.reject(Error(message)); }
  pending.clear();
  worker?.terminate(); worker = null; inFlightPrediction = 0;
  // 釋放已經初始化的舊 MediaPipe（記憶體與影像運算），重新載入時會建新的
  try { holistic?.close?.(); } catch {}
  holistic = null;
  $('start-camera').hidden = false; $('start-camera').disabled = false;
  // 按鈕寬度是固定的（100px），八個字會被裁掉兩端；四個字剛好，錯誤訊息裡也用同一個名字（09-25）
  put('start-camera', '重新載入');
  $('camera-empty').hidden = true;      // 框中央不要同時留著入鏡提示，讓錯誤框自己說明
  status(friendlyMessage(message), true);
}

window.addEventListener('pagehide', stopCamera);
/* 切到別的分頁或別的 App 就關鏡頭——沒有在看的時候不該還在錄。
   手機上這件事很常發生（接個通知就切走了），所以訊息要說清楚怎麼回來，
   不然使用者回來只看到鏡頭黑掉會以為它壞了。 */
/* 切到別的分頁或別的 App 就關鏡頭——沒有在看的時候不該還在錄。
   只看 active：還在下載、根本還沒有鏡頭的階段不要被當成關鏡頭處理
   （下載沒有隱私問題，鏡頭才有），否則按鈕被重設、訊息寫「鏡頭已關閉」，
   而下載監看器還活著，下一個 tick 又把它蓋成「正在下載… 60%」，兩行訊息互相打架。 */
document.addEventListener('visibilitychange', () => {
  if (document.hidden && active) stopCamera('離開頁面時鏡頭已自動關掉');
});


/* 「關於」浮層。內容含教材來源與開源授權，那是引用義務，
   所以是換位置不是刪掉——只是不讓它常駐在主畫面上。 */
(() => {

  const sheet = $('about-sheet'), open = $('about'), close = $('about-close');
  if (!sheet || !open) return;
  /* 用 inert 一次解決焦點與報讀。原本只切 hidden 再把焦點送到關閉鈕，
     沒有圈住焦點也沒有停用背景：從浮層最後一個連結再按 Tab，
     焦點就跑到被遮罩蓋住的「女聲／男聲」「我要比比看」上，
     使用者看不到焦點在哪，也不知道自己已經離開對話框。 */
  const behind = [document.querySelector('.topbar'), document.querySelector('main.layout')];
  const show = on => {
    sheet.hidden = !on;
    behind.forEach(el => { if (el) el.inert = on; });
    if (on) close?.focus(); else open.focus();
  };
  open.addEventListener('click', () => show(sheet.hidden));
  close?.addEventListener('click', () => show(false));
  // 點浮層外面關掉，但點卡片本身不關
  sheet.addEventListener('click', e => { if (e.target === sheet) show(false); });
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && !sheet.hidden) show(false);
  });
})();
