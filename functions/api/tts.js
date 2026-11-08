// 整句語音：GET /api/tts?voice=female|male&w=早安,老師
//
// 為什麼要有這支（2026-09-24，作者的決定）：
//   網站上的句子是使用者當場比出來的，組合無限多，不可能事先錄好每一句。
//   用單詞音檔接起來，每個接縫的語調都是斷的，聽起來一個詞一頓。
//   要跟「一口氣唸完」一樣順，只能在按下「全部唸出」時當場合成整句。
//
// 用的是 Google Cloud 語音合成（臺灣中文 WaveNet：女聲 A、男聲 B），
// 要用自己的 Google Cloud 專案與 API 金鑰（只開語音合成）。這支沒有用量上限，部署的人要自己設預算提醒。
// 網站的單詞音檔（voices/，公開版未附）也是用同樣兩個聲音產生的，整站只有這兩個聲音。
// （2026-09-24：作者比較喜歡微軟的曉臻／雲哲，但 Azure 要綁卡開新帳號；先用 Google 上線試聽。
//   之後要換回微軟，改這支檔案的呼叫與 VOICES、重新產生單詞音檔就好。）
// 金鑰存在 Cloudflare 的秘密變數 GOOGLE_TTS_KEY（只限語音合成使用），網頁與原始碼裡都看不到。
//
// 保護：
//   · 只接受網站那 25 個詞、一句最多 40 個詞（跟網頁句子框的上限 SENTENCE_LIMIT 一樣），別人不能拿去合成任意文字。
//   · 其他網站的網頁來呼叫會被擋（看瀏覽器送的 Sec-Fetch-Site）。
//   · 合成過的句子放進 Cloudflare 快取，同一句不重複花額度。
//   · 這裡出任何錯（沒設金鑰、Google 出錯、額度用完）都回錯誤碼，
//     網頁端會立刻改用單詞音檔唸，使用者不會卡住（見 js/speech.js）。

const WORDS = new Set([
  '一起', '他', '你', '你們', '刷牙', '去', '同學', '多', '好', '學',
  '我', '我們兩個', '早安', '朋友', '洗臉', '玩', '畫畫', '穿衣服',
  '老師', '聊天', '讀書', '起床', '跑步', '運動', '開心',
]);

// 網站上的兩個聲音。單詞音檔（voices/）也是用同樣兩個聲音產生的，整站只有這兩個。
const VOICES = { female: 'cmn-TW-Wavenet-A', male: 'cmn-TW-Wavenet-B' };
const RATE = 0.95;            // 跟單詞音檔一樣慢 5%
// 2026-09-25：原本是 12，但網頁句子框最多放 40 個詞（app.js SENTENCE_LIMIT）。超過 12 個詞時這裡回 400，
// 網頁就改用單詞音檔一個一個唸——作者實測左右手各比一次、句子一長就「又變成一個一個唸」，就是這個原因
//（正式站紀錄：13 個詞起全部 400）。兩邊的上限要一樣。40 個詞大約 100 個字，遠低於 Google 單次 5000 位元組的上限。
const MAX_WORDS = 40;

const fail = (status, text) => new Response(text, {
  status, headers: { 'content-type': 'text/plain; charset=utf-8', 'cache-control': 'no-store' },
});

export async function onRequestGet({ request, env, waitUntil }) {
  // 只給自己的網頁用：瀏覽器從別的網站發出的請求會帶 Sec-Fetch-Site: cross-site
  const site = request.headers.get('sec-fetch-site');
  if (site && site !== 'same-origin' && site !== 'none') return fail(403, 'forbidden');

  const url = new URL(request.url);
  const voice = url.searchParams.get('voice') === 'male' ? 'male' : 'female';
  const words = (url.searchParams.get('w') || '').split(',').filter(Boolean);
  if (!words.length || words.length > MAX_WORDS || !words.every(w => WORDS.has(w))) {
    return fail(400, 'bad request');
  }
  if (!env.GOOGLE_TTS_KEY) return fail(503, 'not configured');

  // 快取鍵只看聲音和詞，網址上其他參數不影響
  const key = new Request(`${url.origin}/api/tts?voice=${voice}&w=${encodeURIComponent(words.join(','))}`);
  const cache = caches.default;
  try {
    const hit = await cache.match(key);
    if (hit) return hit;
  } catch { /* 快取不能用就直接合成 */ }

  let res;
  try {
    res = await fetch(`https://texttospeech.googleapis.com/v1/text:synthesize?key=${env.GOOGLE_TTS_KEY}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        input: { text: words.join('') + '。' },
        voice: { languageCode: 'cmn-TW', name: VOICES[voice] },
        audioConfig: { audioEncoding: 'MP3', speakingRate: RATE, sampleRateHertz: 24000 },
      }),
    });
  } catch {
    return fail(502, 'upstream unreachable');
  }
  if (!res.ok) return fail(502, `upstream ${res.status}`);
  let audio;
  try {
    const { audioContent } = await res.json();
    audio = Uint8Array.from(atob(audioContent), c => c.charCodeAt(0));
  } catch {
    return fail(502, 'bad upstream response');
  }
  if (!audio.byteLength) return fail(502, 'empty audio');

  const out = new Response(audio, {
    headers: {
      'content-type': 'audio/mpeg',
      // 同一句的聲音不會變：瀏覽器和 Cloudflare 都可以放很久
      'cache-control': 'public, max-age=31536000, immutable',
    },
  });
  try { waitUntil(cache.put(key, out.clone())); } catch { /* 快取失敗不影響這次播放 */ }
  return out;
}
