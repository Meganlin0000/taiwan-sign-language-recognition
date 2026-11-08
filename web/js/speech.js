// 唸出辨識結果。整個網站只有兩個聲音：女聲、男聲（Google 臺灣中文 WaveNet A／B），單詞與整句同一個聲音。
// （2026-09-24：作者要整句「一口氣唸完」的順度；接單詞音檔做不到，所以整句改成當場合成。）
//
//   單一個詞 → <audio> 播預先產生的 mp3（voices/<聲音>/NN.mp3，用同樣兩個 Google 聲音預先產生、
//              裁頭尾並統一響度；公開版沒有附音檔，沒有音檔時單詞不出聲、不會報錯）
//              辨識到詞「自動唸出」時改用 Web Audio：手機、平板會擋掉沒在點擊當下開始的 <audio>（見 unlock）
//   整句     → sayAll() 依序試：
//     一、sayCloud：向自己網站的 /api/tts 要整句（functions/api/tts.js 當場用同一個聲音合成）。
//         每認出一個詞就在背景先合成目前的句子（prefetchSentence），按「全部唸出」時通常已經好了。
//         （先合成要有單詞音檔的 manifest.json；公開版沒有，第一次按可能逾時沒聲音，再按一次就會唸。）
//         超過 CLOUD_WAIT_MS 還沒回來、或服務出錯（沒設金鑰、額度用完、斷線）→ 回 null，當下改走下一條。
//     二、sayAllSeamless：用 Web Audio 把單詞音檔排成一條時間軸無縫接起來（不如一的順，但不靠網路服務）。
//     三、sayAllClips：Web Audio 不能用（或舊款 iPhone 會被靜音開關關掉）時，逐個 <audio> 播。
//   最重要的是「使用的時候不出問題」：任何一條失敗都安靜地換下一條，絕不卡住。
//   （公開版沒有音檔，也沒設金鑰時，就不會出聲。）

const GAP_MS = 60;           // 逐個 <audio> 播時，詞與詞之間的間隔
/* 無縫播放時詞與詞之間的空檔（秒）。0 = 完全接在一起。
   音檔頭尾幾乎沒有靜音，所以這個值就是真正聽得到的停頓；覺得黏在一起就調成 0.04 左右。 */
const WORD_GAP_S = 0;
const OVERLAP_S = 0.010;     // 句中版與下一個詞交疊的長度（句中版頭尾已經做了 10 毫秒淡入淡出）
const AUDIO_CACHE_MAX = 64;
const CLOUD_WAIT_MS = 1500;  // 整句合成等這麼久還沒回來，就先用單詞音檔唸（合成仍會在背景完成、下次直接用）
const SENTENCE_CACHE_MAX = 40;  // <audio> 物件上限：一個聲音 25 個詞，全部留著還有餘裕

/* 舊款 iPhone／iPad（Safari 16.4 以前沒有 navigator.audioSession）：Web Audio 會被側邊的靜音開關關掉，
   <audio> 卻不會——單詞有聲、整句沒聲，而且程式偵測不到（狀態照樣是 running）。這種裝置整句直接逐個 <audio> 播。
   iPad 的 Safari 會假裝成 Mac，要用「Mac 卻有觸控」認出來。 */
const IOS = typeof navigator !== 'undefined'
  && (/iP(hone|ad|od)/.test(navigator.userAgent || '')
      || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1));
const WEB_AUDIO_MUTED_BY_SWITCH = IOS && typeof navigator !== 'undefined' && !('audioSession' in navigator);

/* 10 毫秒的無聲 wav：在使用者點擊的當下播它一次，就能把共用的 <audio> 解鎖。
   網站的安全設定（media-src 'self' blob:）不准播 data: 網址，所以播放時轉成 blob: 網址（silentUrl）。 */
const SILENT_WAV_B64 = 'UklGRnQAAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YVAAAACAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgICAgA==';
let silentUrl = '';
function silentWavUrl() {
  if (!silentUrl) {
    const bin = atob(SILENT_WAV_B64);
    const bytes = Uint8Array.from(bin, c => c.charCodeAt(0));
    silentUrl = URL.createObjectURL(new Blob([bytes], { type: 'audio/wav' }));
  }
  return silentUrl;
}

export class Speech {
  constructor(base = 'voices') {
    /* iPhone 開靜音時，Web Audio 預設的 ambient 模式會被一起關掉。設成 playback 跟 <audio> 一致。
       要在任何聲音、任何 AudioContext 之前設一次；播放途中切換，iOS 會亂掉。 */
    try { if ('audioSession' in navigator) navigator.audioSession.type = 'playback'; } catch {}
    this.base = base;
    this.voice = 'female';
    this.enabled = true;
    this.manifest = null;
    this.cache = new Map();      // 單詞的 <audio>
    this._buf = new Map();       // 解碼好的 AudioBuffer（Promise），整句用
    this._bytes = new Map();     // 預先下載、還沒解碼的 mp3（Promise）
    this._warm = new Set();      // 預載過的詞，換聲音時重抓
    this._sent = new Map();      // 整句合成結果（Promise<ArrayBuffer|null>），鍵＝聲音＋詞
    this._pfTimer = 0;
    this._nodes = [];
    this._aborts = new Set();    // 正在等的播放；stop() 讓它們立刻結束
    this.current = null;
    this.sequenceToken = 0;
    this.autoQueue = [];
    this.autoEpoch = 0;
    this.autoDraining = false;
    this.ready = fetch(`${base}/manifest.json`)
      .then(r => (r.ok ? r.json() : null))
      .then(m => { this.manifest = m; return m; })
      .catch(() => null);
  }

  get available() { return !!this.manifest; }

  /* 解鎖聲音。<一定要在使用者點擊、按鍵的當下呼叫>（app.js 在整頁的點擊上掛了這個）。
     為什麼要有（2026-09-25，手機、平板都「辨識到詞不會自動唸」）：
       iPhone／iPad 的瀏覽器只准「使用者點下去的那一刻」開始播聲音，而且每個 <audio> 各算各的。
       自動唸出是辨識到詞的時候才播，那時沒有人在點畫面，每個詞的 <audio> 都會被擋掉、完全沒聲音。
       電腦版的瀏覽器只要使用者點過網頁一次就放行，所以在電腦上測不出來。
     做法：在點擊的當下
       一、建立並叫醒 Web Audio，播一段無聲——之後隨時都能出聲，自動唸出走這條（sayAuto）；
       二、解鎖一個共用的 <audio>——舊款 iPhone 不能用 Web Audio 時，自動唸出改用這一個播。
     可以重複呼叫：已經解鎖就什麼都不做；接完電話、切回瀏覽器被暫停時，下一次點擊會再叫醒。 */
  unlock() {
    if (!WEB_AUDIO_MUTED_BY_SWITCH && this._ctx?.state !== 'running') {
      const ctx = this.ctx;                       // 建立或叫醒
      if (ctx) {
        try {
          const src = ctx.createBufferSource();
          src.buffer = ctx.createBuffer(1, 1, 22050);
          src.connect(ctx.destination);
          src.start(0);
        } catch {}
      }
    }
    if (!this._player) {
      const a = new Audio();
      a.src = silentWavUrl();
      const p = a.play();
      if (p && p.then) p.then(() => { this._player = a; }, () => {});
      else this._player = a;
    }
  }

  get ctx() {
    if (!this._ctx || this._ctx.state === 'closed') {
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return null;
      try { this._ctx = new AC(); } catch { return null; }
    }
    // 不只 suspended：iPhone 接完電話、切回瀏覽器時是 interrupted，也要叫醒
    if (this._ctx.state !== 'running') this._ctx.resume().catch(() => {});
    return this._ctx;
  }

  /** 下載（或拿預載好的）mp3、解碼成 AudioBuffer。單詞版再做保險裁切。失敗回傳 null。 */
  buffer(word, variant = 'word') {
    const url = variant === 'mid' ? this.midUrl(word) : this.url(word);
    if (!url) return Promise.resolve(null);
    const key = `${this.voice}/${variant}/${word}`;
    if (this._buf.has(key)) return this._buf.get(key);
    const ctx = this.ctx;
    if (!ctx) return Promise.resolve(null);
    // 解碼會把位元組拿走（detach），同一份不能解第二次，所以從預載區取出來就刪掉
    const pre = this._bytes.get(key);
    this._bytes.delete(key);
    const p = (async () => {
      const bytes = (pre && await pre) || await fetch(url).then(r => (r.ok ? r.arrayBuffer() : null));
      if (!bytes) return null;
      const raw = await ctx.decodeAudioData(bytes);
      return variant === 'mid' ? raw : this._trim(ctx, raw);
    })().catch(() => null);
    this._buf.set(key, p);
    // 失敗不留在快取裡：網路一時斷掉，恢復後要能重試，不能從此一直退回逐個 <audio>
    p.then(b => { if (!b && this._buf.get(key) === p) this._buf.delete(key); });
    return p;
  }

  /* 單詞版的保險裁切。檔案本身已經裁好（頭尾只剩幾十毫秒、尾巴有淡出），這裡是給瀏覽器還留著
     舊版音檔（頭 0.2 秒、尾 1 秒多靜音）的人用的，不然整句中間會空一拍；也順便去掉 mp3 編碼補的靜音。
     門檻 0.001（-60 dBFS）：原本 0.008 會切掉男聲 100～200 毫秒的尾音，也可能切到 ㄕ、ㄙ、ㄈ 這種很輕的聲母。
     切點不在零的地方會「喀」一聲，所以再加 5 毫秒淡入、20 毫秒淡出。
     句中版不經過這裡——它產生時就剪得很緊、頭尾有淡入淡出。 */
  _trim(ctx, raw) {
    const d = raw.getChannelData(0), sr = raw.sampleRate, thr = 0.001;
    let s = 0, e = d.length - 1;
    while (s < e && Math.abs(d[s]) < thr) s++;
    while (e > s && Math.abs(d[e]) < thr) e--;
    s = Math.max(0, s - Math.round(sr * 0.01));      // 前面留 10 毫秒，不要切到起音
    e = Math.min(d.length - 1, e + Math.round(sr * 0.02));
    const len = e - s + 1;
    const fin = Math.min(len, Math.round(sr * 0.005)), fout = Math.min(len, Math.round(sr * 0.02));
    const out = ctx.createBuffer(raw.numberOfChannels, len, sr);
    for (let ch = 0; ch < raw.numberOfChannels; ch++) {
      const o = out.getChannelData(ch);
      o.set(raw.getChannelData(ch).subarray(s, s + len));
      for (let i = 0; i < fin; i++) o[i] *= i / fin;
      for (let i = 0; i < fout; i++) o[len - 1 - i] *= i / fout;
    }
    return out;
  }

  /* 整串詞排成一條連續的時間軸：先解碼，再用 source.start(絕對時間) 一個接一個排好。
     排程是取樣級精準的，不受 JavaScript 忙碌影響，所以詞與詞之間不會頓。
     回傳 true＝唸完、false＝被 stop() 打斷、null＝這條路不能用（交給逐個 <audio>）。 */
  async sayAllSeamless(words, onIndex) {
    const token = ++this.sequenceToken;     // 一開始就拿號碼牌：下載、解碼途中按停也要算數
    if (WEB_AUDIO_MUTED_BY_SWITCH) return null;
    const ctx = this.ctx;
    if (!ctx) return null;
    const last = words.length - 1;
    // 前面的詞用句中版（沒有句中版就用單詞版），最後一個用單詞版
    const bufs = await Promise.all(words.map(async (w, i) =>
      (i < last && await this.buffer(w, 'mid')) || this.buffer(w)));
    if (token !== this.sequenceToken) return false;     // 載入中被打斷
    if (bufs.some(b => !b)) return null;                 // 有檔解不開
    if (ctx.state !== 'running') {
      // 叫不醒（接電話中、瀏覽器擋自動播放）就交給 <audio>，不要默默不出聲
      await Promise.race([ctx.resume().catch(() => {}), new Promise(r => setTimeout(r, 300))]);
      if (token !== this.sequenceToken) return false;
      if (ctx.state !== 'running') return null;
    }
    this._nodes = [];
    let at = ctx.currentTime + 0.06;                     // 一點點前置，避免第一個字被切
    bufs.forEach((b, i) => {
      const src = ctx.createBufferSource();
      src.buffer = b;
      src.connect(ctx.destination);
      src.start(at);
      this._nodes.push(src);
      const when = at;
      setTimeout(() => { if (token === this.sequenceToken) onIndex?.(i); },
                 Math.max(0, (when - ctx.currentTime) * 1000));
      at += b.duration + (i < last ? WORD_GAP_S - OVERLAP_S : 0);
    });
    await this._sleep(Math.max(0, (at - ctx.currentTime) * 1000));
    if (token !== this.sequenceToken) return false;
    onIndex?.(-1);
    this._nodes = [];
    return true;
  }

  /* 等 ms 毫秒；stop() 會讓它立刻結束。原本被打斷後還要等整句的時間跑完才回傳，按鈕跟著卡好幾秒。 */
  _sleep(ms) {
    return new Promise(resolve => {
      const abort = () => { clearTimeout(t); this._aborts.delete(abort); resolve(false); };
      const t = setTimeout(() => { this._aborts.delete(abort); resolve(true); }, ms);
      this._aborts.add(abort);
    });
  }

  setVoice(voice) {
    if (voice === this.voice) return;
    this.voice = voice;
    this.cache.clear();
    this._buf.clear();
    this._bytes.clear();
    this.stop();
    if (this._warm.size) this.warm([...this._warm]);    // 換了聲音，這一組也先抓好
  }

  setEnabled(on) {
    this.enabled = !!on;
    if (!this.enabled) this.stop();
  }

  url(word) {
    const file = this.manifest?.words?.[word];
    return file ? `${this.base}/${this.voice}/${file}` : null;
  }

  midUrl(word) {
    const file = this.manifest?.mid?.[word];
    return file ? `${this.base}/${this.voice}/${file}` : null;
  }

  /* ── 整句當場合成 ────────────────────────────── */
  sentenceUrl(words) {
    return `/api/tts?voice=${this.voice}&w=${encodeURIComponent(words.join(','))}`;
  }

  _fetchSentence(words) {
    const key = `${this.voice}|${words.join(',')}`;
    if (this._sent.has(key)) return this._sent.get(key);
    const p = fetch(this.sentenceUrl(words))
      .then(r => (r.ok && /audio/.test(r.headers.get('content-type') || '') ? r.arrayBuffer() : null))
      .catch(() => null);
    this._sent.set(key, p);
    if (this._sent.size > SENTENCE_CACHE_MAX) this._sent.delete(this._sent.keys().next().value);
    // 失敗不留：服務恢復之後要能重試
    p.then(b => { if (!b && this._sent.get(key) === p) this._sent.delete(key); });
    return p;
  }

  /* 句子一有變化就在背景先合成。停 0.7 秒才送：比得很快時不會每個詞都打一次。 */
  prefetchSentence(words) {
    clearTimeout(this._pfTimer);
    if (!this.manifest || !words || words.length < 2) return;
    const copy = [...words];
    this._pfTimer = setTimeout(() => this._fetchSentence(copy), 700);
  }

  /* 回傳 true＝唸完、false＝被打斷、null＝這條路現在不能用（交給單詞音檔）。 */
  async sayCloud(words, onIndex) {
    const token = ++this.sequenceToken;
    const got = await Promise.race([
      this._fetchSentence(words),
      new Promise(r => setTimeout(() => r(null), CLOUD_WAIT_MS)),
    ]);
    if (token !== this.sequenceToken) return false;
    if (!got) return null;
    const bytes = got.slice(0);          // 解碼會把位元組拿走，快取裡的那份要留著下次用
    if (WEB_AUDIO_MUTED_BY_SWITCH) {
      // 舊款 iPhone：Web Audio 會被靜音開關關掉，改用 <audio> 播
      const url = URL.createObjectURL(new Blob([bytes], { type: 'audio/mpeg' }));
      const a = new Audio(url);
      onIndex?.(-2);
      const played = await this._play(a);
      URL.revokeObjectURL(url);
      if (token === this.sequenceToken) onIndex?.(-1);
      if (!played) return null;
      return token === this.sequenceToken;
    }
    const ctx = this.ctx;
    if (!ctx) return null;
    let buf;
    // 合成結果頭尾各帶一小段靜音：用跟單詞版同一個保守裁切，唸完詞塊才不會多亮一下
    try { buf = this._trim(ctx, await ctx.decodeAudioData(bytes)); } catch { return null; }
    if (token !== this.sequenceToken) return false;
    if (ctx.state !== 'running') {
      await Promise.race([ctx.resume().catch(() => {}), new Promise(r => setTimeout(r, 300))]);
      if (token !== this.sequenceToken) return false;
      if (ctx.state !== 'running') return null;
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start(ctx.currentTime + 0.03);
    this._nodes = [src];
    onIndex?.(-2);                       // 整句一起唸，詞塊一起亮
    const finished = await this._sleep(buf.duration * 1000 + 60);
    if (token !== this.sequenceToken || !finished) return false;
    onIndex?.(-1);
    this._nodes = [];
    return true;
  }

  audio(word) {
    const url = this.url(word);
    if (!url) return null;
    let a = this.cache.get(url);
    if (!a) {
      a = new Audio(url);
      a.preload = 'auto';
      if (this.cache.size >= AUDIO_CACHE_MAX) this.cache.clear();
      this.cache.set(url, a);
    }
    return a;
  }

  /* 先把用得到的音檔抓下來，第一次唸不用等網路。
     <audio> 的 preload 在 iPhone 上通常不會真的先下載，所以整句要用的 mp3（單詞版與句中版）
     另外用 fetch 抓成位元組放著；解碼要有 AudioContext，留到真的要唸時（只要幾毫秒）。 */
  warm(words) {
    for (const w of words) this._warm.add(w);
    this.ready.then(() => {
      for (const w of words) {
        this.audio(w);
        this._prefetch(w, 'word');
        this._prefetch(w, 'mid');
      }
    });
  }

  _prefetch(word, variant) {
    const url = variant === 'mid' ? this.midUrl(word) : this.url(word);
    const key = `${this.voice}/${variant}/${word}`;
    if (!url || this._buf.has(key) || this._bytes.has(key)) return;
    const p = fetch(url, { priority: 'low' })
      .then(r => (r.ok ? r.arrayBuffer() : null))
      .catch(() => null);
    this._bytes.set(key, p);
    p.then(b => { if (!b && this._bytes.get(key) === p) this._bytes.delete(key); });
  }

  stop({ preserveAutoQueue = false } = {}) {
    if (!preserveAutoQueue) {
      this.autoQueue.length = 0;
      this.autoEpoch++;
      this.autoDraining = false;
    }
    // 先換號碼牌、結束正在等的播放，再取消裝置語音：有些瀏覽器取消時會當場送 onerror，
    // 那時號碼牌若還沒換，正在進行的播放會誤以為「沒開始」而換一條路從頭再唸一次。
    this.sequenceToken++;
    for (const abort of [...this._aborts]) abort();
    this._aborts.clear();
    for (const n of this._nodes) { try { n.stop(); } catch {} }
    this._nodes = [];
    if (this.current) { try { this.current.pause(); this.current.currentTime = 0; } catch {} }
    this.current = null;
  }

  // 確認的詞依序唸完，不打斷前一個詞；所有停止操作會清空佇列。
  enqueue(word) {
    if (!this.enabled) return Promise.resolve(false);
    this.autoQueue.push(word);
    if (this.autoDraining) return Promise.resolve(true);
    this.autoDraining = true;
    const epoch = this.autoEpoch;
    const drain = async () => {
      try {
        while (this.enabled && epoch === this.autoEpoch && this.autoQueue.length) {
          await this.say(this.autoQueue.shift(), { autoEpoch: epoch });
        }
      } finally {
        if (epoch === this.autoEpoch) this.autoDraining = false;
      }
    };
    return drain().catch(() => false);
  }

  /* 從頭播一個 <audio>，等到真的播完才 resolve（呼叫端靠它決定膠囊亮多久）。
     也要聽 pause：stop() 是用 pause() 中斷的，只聽 ended 會永遠等下去。
     回傳 false＝瀏覽器擋下播放（使用者還沒跟網頁互動過）。 */
  async _play(a) {
    this.current = a;
    try {
      a.currentTime = 0;
      await a.play();
    } catch {
      return false;
    }
    if (a.paused) return true;
    await new Promise(resolve => {
      const done = () => {
        a.removeEventListener('ended', done);
        a.removeEventListener('pause', onPause);
        a.removeEventListener('error', done);
        resolve();
      };
      // 自然播完也會先送 pause，等 ended 才算這個詞真正播放完畢。
      const onPause = () => { if (!a.ended) done(); };
      a.addEventListener('ended', done);
      a.addEventListener('pause', onPause);
      a.addEventListener('error', done);
    });
    return true;
  }

  // 播一個詞。force = true 代表使用者主動點的，即使關掉自動唸出也要播。
  async say(word, { force = false, autoEpoch = null } = {}) {
    if (!this.enabled && !force) return false;
    await this.ready;
    if ((!this.enabled && !force) || (autoEpoch !== null && autoEpoch !== this.autoEpoch)) return false;
    const a = this.audio(word);
    if (!a) return false;
    this.stop({ preserveAutoQueue: autoEpoch !== null });
    const token = this.sequenceToken;
    if (autoEpoch !== null) {
      // 自動唸出不在點擊的當下：先走點擊時已經解鎖的 Web Audio（見 unlock）
      const done = await this.sayAuto(word, token);
      if (done !== null) return done;
    }
    let played = await this._play(a);
    if (!played && autoEpoch !== null && this._player && token === this.sequenceToken) {
      // 這個詞的 <audio> 被擋（iPhone／iPad）：改用點擊時解鎖過的那一個 <audio>
      this._player.src = a.src;
      played = await this._play(this._player);
    }
    return played && token === this.sequenceToken;
  }

  /* 用 Web Audio 播一個詞。回傳 true＝唸完、false＝被打斷、null＝這條路現在不能用（改用 <audio>）。
     只用點擊時已經叫醒的那個 AudioContext；不在這裡新建——不在點擊當下新建的，iPhone 上不會出聲。 */
  async sayAuto(word, token) {
    const ctx = this._ctx;
    if (WEB_AUDIO_MUTED_BY_SWITCH || !ctx || ctx.state !== 'running') return null;
    const buf = await this.buffer(word);
    if (token !== this.sequenceToken) return false;
    if (!buf || ctx.state !== 'running') return null;
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start(ctx.currentTime + 0.02);
    this._nodes = [src];
    const finished = await this._sleep(buf.duration * 1000 + 40);
    if (token !== this.sequenceToken || !finished) return false;
    this._nodes = [];
    return true;
  }

  /* 播放整句（三條路見檔案最上面）。
     回傳 true＝唸完、false＝被打斷或播不出來。只有前一條回 null（不能用）才換下一條；
     回 false 代表使用者按了停，不能再換一條路從頭唸。 */
  async sayAll(words, onIndex) {
    this.stop();
    // 趁還在使用者點擊的當下建立／叫醒 Web Audio：Safari 只在點擊當下才肯讓它出聲
    if (words.length && !WEB_AUDIO_MUTED_BY_SWITCH) void this.ctx;
    await this.ready;
    if (words.length > 1) {
      const cloud = await this.sayCloud(words, onIndex);            // 一、整句當場合成
      if (cloud !== null) return cloud;
    }
    const ok = await this.sayAllSeamless(words, onIndex);           // 二、單詞音檔無縫接續
    if (ok !== null) return ok;
    return this.sayAllClips(words, onIndex);                        // 三、Web Audio 不能用時的退路
  }

  // 逐個 <audio> 播放（Web Audio 不能用時的退路），詞與詞之間停 GAP_MS。
  async sayAllClips(words, onIndex) {
    await this.ready;
    this.stop();
    const token = this.sequenceToken;
    for (let i = 0; i < words.length; i++) {
      const a = this.audio(words[i]);
      if (!a) continue;
      onIndex?.(i);
      if (!(await this._play(a)) || token !== this.sequenceToken) return false;
      if (i < words.length - 1 && !(await this._sleep(GAP_MS))) return false;
    }
    if (token !== this.sequenceToken) return false;
    onIndex?.(-1);
    this.current = null;
    return true;
  }
}
