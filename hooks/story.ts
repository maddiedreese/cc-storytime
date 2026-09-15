/**
 * One story at a time: prefill the seed's prompt, then a token per `step`
 * until EOS, the token cap or the context's end. The hooks module ticks it
 * from a `$.clock.every` so no single hook call runs long, and reads `text`
 * for the band.
 */
import { EOS, BOS, decode, encode, type Tokenizer } from './tokenizer.ts'
import { TinyModel, rngOf, sample } from './model.ts'

export type StoryOptions = {
  readonly maxTokens?: number
  readonly temperature?: number
  readonly topK?: number
  readonly seed?: number
}

export class Story {
  readonly prompt: string
  text: string
  done = false
  /** Tokens generated so far, past the prompt. */
  generated = 0
  /** Milliseconds spent inside the model, for the footer's tok/s. */
  modelMs = 0

  private readonly tokens: number[]
  private pos = 0
  private token = BOS
  private readonly random: () => number
  private readonly maxTokens: number
  private readonly temperature: number
  private readonly topK: number

  constructor(
    private readonly model: TinyModel,
    private readonly tokenizer: Tokenizer,
    prompt: string,
    options: StoryOptions = {},
  ) {
    this.prompt = prompt
    this.text = prompt
    this.tokens = encode(tokenizer, prompt)
    this.random = rngOf(options.seed ?? 1)
    this.maxTokens = options.maxTokens ?? 60
    this.temperature = options.temperature ?? 0.85
    this.topK = options.topK ?? 40
  }

  /** Runs the whole prompt through the model; a few milliseconds. */
  prefill(): void {
    const start = performance.now()
    while (this.pos < this.tokens.length) {
      this.model.forward(this.token, this.pos)
      this.token = this.tokens[this.pos]!
      this.pos++
    }
    this.modelMs += performance.now() - start
  }

  /** Generates up to `count` more tokens; returns whether the text changed. */
  step(count = 1): boolean {
    if (this.done) return false
    if (this.pos < this.tokens.length) this.prefill()
    let changed = false
    const start = performance.now()
    for (let i = 0; i < count && !this.done; i++) {
      if (this.pos >= this.model.maxContext || this.generated >= this.maxTokens) {
        this.finish()
        break
      }
      const logits = this.model.forward(this.token, this.pos)
      const next = sample(logits, this.random, this.temperature, this.topK)
      this.pos++
      this.generated++
      if (next === EOS) {
        this.finish()
        break
      }
      this.text += decode(this.tokenizer, this.token, next)
      this.token = next
      changed = true
      if (this.generated >= 24 && /[.!?]$/.test(this.text)) {
        this.finish()
        break
      }
    }
    this.modelMs += performance.now() - start
    return changed || this.done
  }

  private finish() {
    this.done = true
    this.text = this.text.trimEnd()
    if (!/[.!?"]$/.test(this.text)) this.text += '…'
  }

  get tokensPerSecond(): number {
    const total = this.tokens.length + this.generated
    return this.modelMs > 0 ? (total * 1000) / this.modelMs : 0
  }
}
