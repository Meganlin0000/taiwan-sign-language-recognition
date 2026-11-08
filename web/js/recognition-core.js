export const FRAME_COUNT = 80;
/* FEATURE_COUNT 是「逐幀從 MediaPipe 擷取」的維度，永遠是 258。
   模型實際吃的是 291 維（多 33 維指向幾何），那是 preprocess.js 在送進模型前才加上去的。
   兩個數字分開寫，免得有人以為改這裡就能換模型。 */
export const FEATURE_COUNT = 258;
export const MODEL_FEATURE_COUNT = 291;

export function extractKeypoints(result) {
  const output = new Float32Array(FEATURE_COUNT);
  let offset = 0;
  for (const [key, count, coordinates] of [['poseLandmarks', 33, ['x','y','z','visibility']], ['leftHandLandmarks',21,['x','y','z']], ['rightHandLandmarks',21,['x','y','z']]]) {
    const landmarks = result[key];
    if (landmarks?.length === count) {
      for (const point of landmarks) for (const name of coordinates) output[offset++] = Number.isFinite(point[name]) ? point[name] : 0;
    } else offset += count * coordinates.length;
  }
  return output;
}

export function validFraming(result) {
  return result.poseLandmarks?.length === 33 && (result.leftHandLandmarks?.length === 21 || result.rightHandLandmarks?.length === 21);
}

export function rankScores(scores, labels) {
  if (scores.length !== 25 || scores.some(x => !Number.isFinite(x))) throw new Error('Invalid prediction');
  return scores.map((score,id)=>({id,label:labels[String(id)],score})).sort((a,b)=>b.score-a.score);
}
