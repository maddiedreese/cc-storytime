/**
 * The llama2.c BPE tokenizer (tok512.bin): pieces with scores, greedy
 * merges by score, byte fallback for what no piece covers.
 */

export const BOS = 1
export const EOS = 2

export type Tokenizer = {
  readonly vocab: readonly string[]
  readonly scores: Float32Array
  readonly ids: ReadonlyMap<string, number>
}

const utf8 = new TextDecoder()

/**
 * Parses the tokenizer file: an int32 max piece length, then per token a
 * float32 score, an int32 length and that many bytes of piece.
 */
export function tokenizerOf(bytes: Uint8Array): Tokenizer {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const vocab: string[] = []
  const scoreList: number[] = []
  let at = 4
  while (at < bytes.byteLength) {
    const score = view.getFloat32(at, true)
    const length = view.getInt32(at + 4, true)
    at += 8
    vocab.push(utf8.decode(bytes.subarray(at, at + length)))
    scoreList.push(score)
    at += length
  }
  const ids = new Map<string, number>()
  vocab.forEach((piece, id) => {
    if (!ids.has(piece)) ids.set(piece, id)
  })
  return { vocab, scores: Float32Array.from(scoreList), ids }
}

/**
 * Encodes text the way llama2.c's `encode` does: a leading space as
 * sentencepiece adds, one token per codepoint (bytes as `<0xXX>` where
 * no piece matches), then the best-scoring adjacent merge until none is
 * left. BOS is not added; the caller decides.
 */
export function encode(t: Tokenizer, text: string, dummyPrefix = true): number[] {
  const tokens: number[] = []
  const source = dummyPrefix && text.length > 0 ? ' ' + text : text
  for (const char of source) {
    const id = t.ids.get(char)
    if (id !== undefined) {
      tokens.push(id)
      continue
    }
    for (const byte of new TextEncoder().encode(char)) tokens.push(byte + 3)
  }
  for (;;) {
    let bestScore = -1e10
    let bestId = -1
    let bestAt = -1
    for (let i = 0; i + 1 < tokens.length; i++) {
      const merged = t.vocab[tokens[i]!]! + t.vocab[tokens[i + 1]!]!
      const id = t.ids.get(merged)
      if (id !== undefined && t.scores[id]! > bestScore) {
        bestScore = t.scores[id]!
        bestId = id
        bestAt = i
      }
    }
    if (bestAt < 0) break
    tokens.splice(bestAt, 2, bestId)
  }
  return tokens
}

const BYTE_PIECE = /^<0x([0-9A-Fa-f]{2})>$/

/**
 * The text of one token after `previous`: the piece, less the space a
 * sentencepiece piece carries straight after BOS, and a raw byte for a
 * `<0xXX>` piece.
 */
export function decode(t: Tokenizer, previous: number, token: number): string {
  let piece = t.vocab[token] ?? ''
  if (previous === BOS && piece.startsWith(' ')) piece = piece.slice(1)
  const byte = BYTE_PIECE.exec(piece)
  return byte ? String.fromCharCode(parseInt(byte[1]!, 16)) : piece
}
