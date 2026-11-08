/* 模型推論放在背景執行緒，不卡畫面；影像與資料都不會離開這支程式。
   2026-09-23 換成新模型：輸入 [1, 80, 291]（MediaPipe 關鍵點 + 33 維指向幾何）。
   前處理在主執行緒的 preprocess.js 做完才送過來，這裡只負責推論。 */
const ROWS = 80, DIM = 291, LEN = ROWS * DIM;   // 23280
importScripts('../vendor/tfjs/tf.min.js');
let model;
let initialization;
/* 類別先驗校正。模型訓練集裡各詞的樣本數不一樣（3.12% 到 5.64%），
   輸入一模糊就會往樣本多的那一類倒：兩個詞機率很接近時，答案會偏向樣本多的那一個；
   模型很有把握時，除完先驗也不會換答案。
   校正方式：輸出機率除以先驗再正規化（α=1，等於測試時不假設訓練集的類別比例）。
   在考卷 D 上試過：正確率略升、沒有任何一個詞變差（單一種子，D 參與過設定選擇，不能當成績；成績見 README）。
   α 是用原理定的不是挑的：驗證集在 98.90% 的天花板上分不出差異。
   讀不到先驗檔就不做校正，不要讓它變成單點故障。 */
let prior = null;
function initialize() {
  if (!initialization) initialization = (async () => {
    await tf.setBackend('cpu');
    await tf.ready();
    // cache: 'no-cache' = 每次都跟伺服器確認有沒有變（條件式請求，沒變就不重抓）。
    // 模型檔的路徑沒有版本號，換了新模型之後瀏覽器會繼續用快取裡的舊 model.json，
    // 然後在這裡報 expected input_layer to have shape [null,80,258]——實測踩到過。
    model = await tf.loadLayersModel('../model/model.json', { requestInit: { cache: 'no-cache' } });
    try {
      const r = await fetch('../model/class-prior.json', { cache: 'no-cache' });
      if (r.ok) {
        const d = await r.json();
        if (Array.isArray(d.先驗) && d.先驗.length === 25 && d.先驗.every(v => v > 0)) {
          prior = d.先驗;
        }
      }
    } catch { /* 沒有先驗檔就照常跑 */ }
    const warm = tf.zeros([1,ROWS,DIM]);
    const result = model.predict(warm);
    await result.data(); result.dispose(); warm.dispose();
  })();
  return initialization;
}
self.onmessage = async ({data}) => {
  try {
    await initialize();
    if (data.type === 'init') { self.postMessage({id:data.id,type:'ready'}); return; }
    // 允許 1 筆或 2 筆。2 筆＝測試時鏡像：原始與左右鏡像各一份，
    // 當成一個 batch 一次推論再把機率平均。比分兩次送快，也少一次跨執行緒往返。
    const n = data.values instanceof Float32Array ? data.values.length / LEN : 0;
    if (data.type !== 'predict' || (n !== 1 && n !== 2) || !data.values.every(Number.isFinite)) throw new Error('Invalid input');
    const start = performance.now();
    const input = tf.tensor3d(data.values,[n,ROWS,DIM]);
    let result;
    try {
      result = model.predict(input);
      const flat = Array.from(await result.data());
      let scores = n === 1 ? flat
        : Array.from({length:25}, (_,k) => (flat[k] + flat[25+k]) / 2);
      if (prior) {
        const q = scores.map((v, k) => v / prior[k]);
        const sum = q.reduce((a, b) => a + b, 0);
        if (sum > 0 && Number.isFinite(sum)) scores = q.map(v => v / sum);
      }
      if (scores.length !== 25 || scores.some(x=>!Number.isFinite(x))) throw new Error('Invalid output');
      self.postMessage({id:data.id,type:'prediction',scores,milliseconds:performance.now()-start});
    } finally { result?.dispose(); input.dispose(); }
  } catch (error) { self.postMessage({id:data?.id,type:'error',message:error.message}); }
};
