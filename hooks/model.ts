/**
 * A llama2.c-shaped transformer (TinyStories-260K) over the int8 pack
 * tools/pack.py writes: RMSNorm, RoPE, grouped-query attention with a KV
 * cache, SwiGLU, tied embeddings. Float32 math on typed arrays; weights are
 * dequantized once at load. One `forward(token, pos)` is one token.
 */

export type Config = {
  readonly dim: number
  readonly hidden: number
  readonly layers: number
  readonly heads: number
  readonly kvHeads: number
  readonly vocab: number
  readonly seqLen: number
}

const RMS_EPS = 1e-5
const ROPE_THETA = 10000

type Reader = {
  f32: (count: number) => Float32Array
  q8: (rows: number, cols: number) => Float32Array
}

function readerOf(bytes: Uint8Array): Reader {
  let at = 0
  const f32 = (count: number): Float32Array => {
    const out = new Float32Array(count)
    const view = new DataView(bytes.buffer, bytes.byteOffset + at, count * 4)
    for (let i = 0; i < count; i++) out[i] = view.getFloat32(i * 4, true)
    at += count * 4
    return out
  }
  const q8 = (rows: number, cols: number): Float32Array => {
    const q = new Int8Array(bytes.buffer, bytes.byteOffset + at, rows * cols)
    at += rows * cols
    while (at % 4) at++
    const scales = f32(rows)
    const out = new Float32Array(rows * cols)
    for (let r = 0; r < rows; r++) {
      const s = scales[r]!
      const base = r * cols
      for (let c = 0; c < cols; c++) out[base + c] = q[base + c]! * s
    }
    return out
  }
  return { f32, q8 }
}

export class TinyModel {
  readonly config: Config
  private readonly tokEmb: Float32Array
  private readonly rmsAtt: Float32Array
  private readonly wq: Float32Array
  private readonly wk: Float32Array
  private readonly wv: Float32Array
  private readonly wo: Float32Array
  private readonly rmsFfn: Float32Array
  private readonly w1: Float32Array
  private readonly w2: Float32Array
  private readonly w3: Float32Array
  private readonly rmsFinal: Float32Array
  private readonly ropeCos: Float32Array
  private readonly ropeSin: Float32Array

  private readonly x: Float32Array
  private readonly xb: Float32Array
  private readonly xb2: Float32Array
  private readonly hb: Float32Array
  private readonly hb2: Float32Array
  private readonly q: Float32Array
  private readonly k: Float32Array
  private readonly v: Float32Array
  private readonly att: Float32Array
  private readonly logits: Float32Array
  private readonly keyCache: Float32Array
  private readonly valueCache: Float32Array

  readonly maxContext: number

  constructor(bytes: Uint8Array, maxContext = 128) {
    const header = new DataView(bytes.buffer, bytes.byteOffset, 28)
    const [dim, hidden, layers, heads, kvHeads, vocab, seqLen] = Array.from(
      { length: 7 },
      (_, i) => header.getInt32(i * 4, true),
    ) as [number, number, number, number, number, number, number]
    this.config = { dim, hidden, layers, heads, kvHeads, vocab, seqLen }
    this.maxContext = Math.min(maxContext, seqLen)

    const kvDim = (dim * kvHeads) / heads
    const headSize = dim / heads
    const read = readerOf(bytes.subarray(28))
    this.tokEmb = read.q8(vocab, dim)
    this.rmsAtt = read.f32(layers * dim)
    this.wq = read.q8(layers * dim, dim)
    this.wk = read.q8(layers * kvDim, dim)
    this.wv = read.q8(layers * kvDim, dim)
    this.wo = read.q8(layers * dim, dim)
    this.rmsFfn = read.f32(layers * dim)
    this.w1 = read.q8(layers * hidden, dim)
    this.w2 = read.q8(layers * dim, hidden)
    this.w3 = read.q8(layers * hidden, dim)
    this.rmsFinal = read.f32(dim)

    const half = headSize / 2
    this.ropeCos = new Float32Array(this.maxContext * half)
    this.ropeSin = new Float32Array(this.maxContext * half)
    for (let pos = 0; pos < this.maxContext; pos++) {
      for (let i = 0; i < half; i++) {
        const freq = 1 / Math.pow(ROPE_THETA, (2 * i) / headSize)
        this.ropeCos[pos * half + i] = Math.cos(pos * freq)
        this.ropeSin[pos * half + i] = Math.sin(pos * freq)
      }
    }

    this.x = new Float32Array(dim)
    this.xb = new Float32Array(dim)
    this.xb2 = new Float32Array(dim)
    this.hb = new Float32Array(hidden)
    this.hb2 = new Float32Array(hidden)
    this.q = new Float32Array(dim)
    this.k = new Float32Array(kvDim)
    this.v = new Float32Array(kvDim)
    this.att = new Float32Array(heads * this.maxContext)
    this.logits = new Float32Array(vocab)
    this.keyCache = new Float32Array(layers * this.maxContext * kvDim)
    this.valueCache = new Float32Array(layers * this.maxContext * kvDim)
  }

  /**
   * One step: the logits for the token after `token` at position `pos`.
   * Positions run 0..maxContext-1; the KV cache is what earlier calls left.
   */
  forward(token: number, pos: number): Float32Array {
    const { dim, hidden, layers, heads, kvHeads, vocab } = this.config
    const kvDim = (dim * kvHeads) / heads
    const kvMul = heads / kvHeads
    const headSize = dim / heads
    const half = headSize / 2
    const { x, xb, xb2, hb, hb2, q, k, v, att } = this
    const ctx = this.maxContext
    if (pos >= ctx) throw new Error(`position ${pos} outside context ${ctx}`)

    x.set(this.tokEmb.subarray(token * dim, (token + 1) * dim))

    for (let l = 0; l < layers; l++) {
      rmsnorm(xb, x, this.rmsAtt, l * dim, dim)
      matmul(q, xb, this.wq, l * dim * dim, dim, dim)
      matmul(k, xb, this.wk, l * kvDim * dim, dim, kvDim)
      matmul(v, xb, this.wv, l * kvDim * dim, dim, kvDim)

      for (let i = 0; i < dim; i += 2) {
        const pair = (i % headSize) / 2
        const c = this.ropeCos[pos * half + pair]!
        const s = this.ropeSin[pos * half + pair]!
        const q0 = q[i]!
        const q1 = q[i + 1]!
        q[i] = q0 * c - q1 * s
        q[i + 1] = q0 * s + q1 * c
        if (i < kvDim) {
          const k0 = k[i]!
          const k1 = k[i + 1]!
          k[i] = k0 * c - k1 * s
          k[i + 1] = k0 * s + k1 * c
        }
      }

      const loff = l * ctx * kvDim
      this.keyCache.set(k, loff + pos * kvDim)
      this.valueCache.set(v, loff + pos * kvDim)

      for (let h = 0; h < heads; h++) {
        const qOff = h * headSize
        const attOff = h * ctx
        const kvHead = Math.floor(h / kvMul) * headSize
        for (let t = 0; t <= pos; t++) {
          const kOff = loff + t * kvDim + kvHead
          let score = 0
          for (let i = 0; i < headSize; i++) score += q[qOff + i]! * this.keyCache[kOff + i]!
          att[attOff + t] = score / Math.sqrt(headSize)
        }
        softmax(att, attOff, pos + 1)
        xb.fill(0, qOff, qOff + headSize)
        for (let t = 0; t <= pos; t++) {
          const vOff = loff + t * kvDim + kvHead
          const a = att[attOff + t]!
          for (let i = 0; i < headSize; i++) xb[qOff + i] = xb[qOff + i]! + a * this.valueCache[vOff + i]!
        }
      }

      matmul(xb2, xb, this.wo, l * dim * dim, dim, dim)
      for (let i = 0; i < dim; i++) x[i] = x[i]! + xb2[i]!

      rmsnorm(xb, x, this.rmsFfn, l * dim, dim)
      matmul(hb, xb, this.w1, l * hidden * dim, dim, hidden)
      matmul(hb2, xb, this.w3, l * hidden * dim, dim, hidden)
      for (let i = 0; i < hidden; i++) {
        const val = hb[i]!
        hb[i] = (val / (1 + Math.exp(-val))) * hb2[i]!
      }
      matmul(xb, hb, this.w2, l * dim * hidden, hidden, dim)
      for (let i = 0; i < dim; i++) x[i] = x[i]! + xb[i]!
    }

    rmsnorm(x, x, this.rmsFinal, 0, dim)
    matmul(this.logits, x, this.tokEmb, 0, dim, vocab)
    return this.logits
  }
}

function rmsnorm(out: Float32Array, x: Float32Array, w: Float32Array, wOff: number, n: number) {
  let ss = 0
  for (let j = 0; j < n; j++) ss += x[j]! * x[j]!
  const scale = 1 / Math.sqrt(ss / n + RMS_EPS)
  for (let j = 0; j < n; j++) out[j] = w[wOff + j]! * (scale * x[j]!)
}

/** out[i] = sum_j w[wOff + i*n + j] * x[j], for i in 0..d */
function matmul(out: Float32Array, x: Float32Array, w: Float32Array, wOff: number, n: number, d: number) {
  for (let i = 0; i < d; i++) {
    let val = 0
    const row = wOff + i * n
    for (let j = 0; j < n; j++) val += w[row + j]! * x[j]!
    out[i] = val
  }
}

function softmax(x: Float32Array, off: number, n: number) {
  let max = x[off]!
  for (let i = 1; i < n; i++) if (x[off + i]! > max) max = x[off + i]!
  let sum = 0
  for (let i = 0; i < n; i++) {
    const e = Math.exp(x[off + i]! - max)
    x[off + i] = e
    sum += e
  }
  for (let i = 0; i < n; i++) x[off + i] = x[off + i]! / sum
}

/** A small deterministic PRNG (mulberry32), so a story can be replayed from its seed. */
export function rngOf(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

export function argmax(logits: Float32Array): number {
  let best = 0
  for (let i = 1; i < logits.length; i++) if (logits[i]! > logits[best]!) best = i
  return best
}

/**
 * Samples a token: temperature, then top-k over the softmax. `temperature`
 * 0 is argmax. Mutates `logits`.
 */
export function sample(
  logits: Float32Array,
  random: () => number,
  temperature = 0.9,
  topK = 40,
): number {
  if (temperature <= 0) return argmax(logits)
  for (let i = 0; i < logits.length; i++) logits[i] = logits[i]! / temperature
  softmax(logits, 0, logits.length)
  const order = Array.from(logits.keys()).sort((a, b) => logits[b]! - logits[a]!)
  const kept = order.slice(0, Math.max(1, topK))
  let total = 0
  for (const id of kept) total += logits[id]!
  let r = random() * total
  for (const id of kept) {
    r -= logits[id]!
    if (r <= 0) return id
  }
  return kept[kept.length - 1]!
}

export function bytesOfBase64(b64: string): Uint8Array {
  const binary = atob(b64)
  const out = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i)
  return out
}
