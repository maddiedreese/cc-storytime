import type { On, RenderElement, Timer } from 'claude-code'

import { TinyModel, bytesOfBase64 } from './model.ts'
import { IDLE_NAMES, seedOf, type Activity, type Seed } from './seed.ts'
import { Story } from './story.ts'
import { tokenizerOf, type Tokenizer } from './tokenizer.ts'
import { bandView } from './views.tsx'
import { MODEL_B64, TOKENIZER_B64 } from './weights.ts'

/** Milliseconds between generation ticks; two tokens a tick reads as typing. */
const TICK_MS = 80
const TOKENS_PER_TICK = 2
/** A tool call starts a fresh story only this long after the last one began. */
const COOLDOWN_MS = 12_000
const COMMAND_NAME = 'story'
const STORE_ENABLED_KEY = 'enabled'

/**
 * The engine as `session.start` bound it: the calls later hooks, the ticker
 * and the command make on `$`.
 */
type Host = {
  now: () => Promise<number>
  every: (ms: number, fn: () => void) => Timer
  invalidate: () => void
  storeGet: (key: string) => Promise<unknown>
  storeSet: (key: string, value: unknown) => Promise<void>
  uiLog: (text: string) => void
  registerCommand: (spec: { name: string; description: string; argumentHint?: string }) => Promise<unknown>
}

/**
 * Registers the mod: a story above the prompt from a local TinyStories
 * transformer, reseeded by the person's prompt and by the tools Claude
 * calls, and `/story` to restart, name or hide it.
 *
 * @param on the engine's registrar
 */
export function register(on: On) {
  let host: Host | null = null
  let model: TinyModel | null = null
  let tokenizer: Tokenizer | null = null
  let story: Story | null = null
  let seed: Seed | null = null
  let ticker: Timer | null = null
  let isEnabled = true
  let startedAtMs = 0
  let storiesBegun = 0

  function engine(): { model: TinyModel; tokenizer: Tokenizer } {
    model ??= new TinyModel(bytesOfBase64(MODEL_B64))
    tokenizer ??= tokenizerOf(bytesOfBase64(TOKENIZER_B64))
    return { model, tokenizer }
  }

  function begin(activity: Activity | null, name?: string) {
    if (!host) return
    const { model: m, tokenizer: t } = engine()
    storiesBegun += 1
    const fallback = name ?? IDLE_NAMES[storiesBegun % IDLE_NAMES.length]!
    seed = seedOf(activity, fallback)
    if (name) seed = { ...seed, prompt: seed.prompt.replaceAll(seed.name, name), name }
    story = new Story(m, t, seed.prompt, { seed: hashOf(`${seed.prompt}#${storiesBegun}`) })
    story.prefill()
    host.invalidate()
    startTicker(host)
  }

  function startTicker(bound: Host) {
    ticker?.cancel()
    ticker = bound.every(TICK_MS, () => {
      if (!story || story.done) {
        ticker?.cancel()
        ticker = null
        return
      }
      if (story.step(TOKENS_PER_TICK)) bound.invalidate()
    })
  }

  async function maybeBegin(activity: Activity) {
    if (!host || !isEnabled) return
    const now = await host.now()
    const isFresh = story !== null && !story.done && now - startedAtMs < COOLDOWN_MS
    const isRecent = now - startedAtMs < COOLDOWN_MS
    if (isFresh || (isRecent && story?.done)) return
    startedAtMs = now
    begin(activity)
  }

  on('session.start', async ($, e, next) => {
    host = {
      now: () => $.clock.now(),
      every: (ms, fn) => $.clock.every(ms, fn),
      invalidate: () => $.ui.invalidate('ui.render'),
      storeGet: key => $.store.get(key),
      storeSet: (key, value) => $.store.set(key, value),
      uiLog: text => $.ui.log(text),
      registerCommand: spec => $.command.register(spec),
    }
    const stored = await host.storeGet(STORE_ENABLED_KEY).catch(() => undefined)
    isEnabled = stored !== false
    await host
      .registerCommand({
        name: COMMAND_NAME,
        description: 'Storytime: a new story above the prompt, /story <name>, /story off, /story on',
        argumentHint: '[name | on | off]',
      })
      .catch(() => undefined)
    if (e.isInteractive && isEnabled) {
      startedAtMs = await host.now()
      begin(null)
    }
    return next(e)
  })

  on('turn.start', ($, e, next) => {
    if (e.text.trim() !== '') void maybeBegin({ tool: 'prompt', text: e.text })
    return next(e)
  })

  on('tool.call', async ($, e, next) => {
    try {
      return await next(e)
    } finally {
      const fields = e as unknown as Record<string, unknown>
      void maybeBegin({
        tool: e.tool,
        path: stringOf(fields.file_path) ?? stringOf(fields.notebook_path) ?? stringOf(fields.path),
        command: stringOf(fields.command),
      })
    }
  })

  on('ui.render', { component: 'AbovePrompt' }, async ($, e, next) => {
    const below = await next(e)
    if (!isEnabled || !story || !seed || e.props.hasSurvey || e.surface !== 'terminal') {
      return below
    }
    const { Box, Text } = $.ui.resolve(e)
    const band = bandView(
      { Box, Text },
      {
        text: story.text,
        done: story.done,
        because: `seeded by ${seed.because}`,
        tokensPerSecond: story.tokensPerSecond,
        maxRows: e.props.maxRows,
        columns: e.props.bodyColumns,
      },
    )
    return stack(Box, below, band)
  })

  on('command.run', { command: COMMAND_NAME }, async ($, e, next) => {
    if (!host) return next(e)
    const args = e.args.trim()
    if (args === 'off') {
      isEnabled = false
      ticker?.cancel()
      ticker = null
      await host.storeSet(STORE_ENABLED_KEY, false).catch(() => undefined)
      host.invalidate()
      return { text: 'Storytime hidden. /story on brings it back.' }
    }
    if (args === 'on') {
      isEnabled = true
      await host.storeSet(STORE_ENABLED_KEY, true).catch(() => undefined)
    }
    const name = args === 'on' || args === '' ? undefined : args.split(/\s+/)[0]
    startedAtMs = await host.now()
    begin(null, name && name[0]!.toUpperCase() + name.slice(1))
    return { text: `A new story about ${seed?.name ?? 'someone'} is being written above the prompt.` }
  })
}

/** The band's tree over whatever the hooks beneath drew there. */
function stack(
  Box: (props: { flexDirection: 'column'; children?: RenderElement[] }) => RenderElement,
  below: RenderElement,
  band: RenderElement,
): RenderElement {
  return Box({ flexDirection: 'column', children: [below, band] })
}

function stringOf(value: unknown): string | undefined {
  return typeof value === 'string' && value !== '' ? value : undefined
}

/** A 32-bit FNV-1a hash, so the sampler's seed follows the prompt. */
function hashOf(text: string): number {
  let hash = 0x811c9dc5
  for (let i = 0; i < text.length; i++) {
    hash ^= text.charCodeAt(i)
    hash = Math.imul(hash, 0x01000193)
  }
  return hash >>> 0
}
