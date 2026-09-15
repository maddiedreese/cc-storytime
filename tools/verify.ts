// Runs the TypeScript port under Node against the C reference's greedy
// output for the same prompt. `node tools/verify.ts`
import { MODEL_B64, TOKENIZER_B64 } from '../hooks/weights.ts'
import { TinyModel, argmax, bytesOfBase64, rngOf, sample } from '../hooks/model.ts'
import { BOS, decode, encode, tokenizerOf } from '../hooks/tokenizer.ts'

const t0 = performance.now()
const model = new TinyModel(bytesOfBase64(MODEL_B64))
const tok = tokenizerOf(bytesOfBase64(TOKENIZER_B64))
console.log(`loaded in ${(performance.now() - t0).toFixed(1)} ms`, model.config)

function run(prompt: string, steps: number, greedy: boolean, dummyPrefix: boolean): string {
  const promptTokens = encode(tok, prompt, dummyPrefix)
  const random = rngOf(1)
  let token = BOS
  let out = ''
  for (let pos = 0; pos < steps; pos++) {
    const logits = model.forward(token, pos)
    let next: number
    if (pos < promptTokens.length) next = promptTokens[pos]!
    else {
      next = greedy ? argmax(logits) : sample(logits, random)
      out += decode(tok, token, next)
    }
    token = next
  }
  return out
}

const t1 = performance.now()
const greedy = run('Once upon a time there was a', 32, true, false)
const ms = performance.now() - t1
console.log('C ref  : little girl named Lily. She loved to play outside in the park.')
console.log('TS port:', greedy)
console.log(`32 steps in ${ms.toFixed(1)} ms = ${(32000 / ms).toFixed(0)} tok/s (first run, cold JIT)`)

const t2 = performance.now()
const N = 100
for (let i = 0; i < N; i++) model.forward(i % 512, i % 100)
const per = (performance.now() - t2) / N
console.log(`steady state: ${per.toFixed(3)} ms/token = ${(1000 / per).toFixed(0)} tok/s`)

console.log('sampled:', run('Once upon a time, there was a little robot named Auth.', 48, false, true))
console.log('tokens :', JSON.stringify(encode(tok, 'Once upon a time, there was a little robot named Auth.').map(i => tok.vocab[i])))
