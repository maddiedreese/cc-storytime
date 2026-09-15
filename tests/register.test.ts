import { describe, expect, mock, test } from 'claude-code/testing'
import type { On, RenderElement } from 'claude-code'

const BAND = {
  surface: 'terminal',
  component: 'AbovePrompt',
  requestId: 'band',
  props: {
    hasSurvey: false,
    isWorking: true,
    maxRows: 10,
    bodyColumns: 80,
    scroll: { offset: 0, bodyRows: 9 },
    view: {},
  },
} as const

/** Every string in a tree, joined: what the band would show. */
function textOf(node: unknown): string {
  if (typeof node === 'string') return node
  if (Array.isArray(node)) return node.map(textOf).join('')
  if (node && typeof node === 'object') {
    const record = node as { props?: { children?: unknown }; children?: unknown }
    return textOf(record.children ?? record.props?.children ?? '')
  }
  return ''
}

/** The world beneath the mod: a session, a clock, a store, an empty band. */
function seat(on: On) {
  const clock = mock.clock(on)
  mock.store(on, {})
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('command.register', ($, e) => ({ value: { command: e.name } }))
  on('turn.start', ($, e) => ({ turnId: e.turnId }))
  on('ui.invalidate', () => ({ value: undefined }))
  on('ui.log', () => ({ value: undefined }))
  on('ui.render', ($, e) => {
    const { Box } = $.ui.resolve(e)
    return Box({}) as RenderElement
  })
  return clock
}

describe('register', () => {
  test('an edit seeds a story whose hero is named for the file', async ($, on) => {
    const clock = seat(on)
    on('tool.call', () => ({ result: 'ok', text: 'ok' }))

    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work' })
    await clock.advance(20_000)
    await $.tool.call({
      tool: 'Edit',
      file_path: '/work/src/auth-service.ts',
      old_string: 'a',
      new_string: 'b',
    })
    await clock.settle()
    const early = textOf(await $.ui.render(BAND))
    expect(early).toContain('there was a little pencil named Auth.')
    expect(early).toContain('seeded by Edit auth-service.ts')
    expect(early).toContain('▍')

    await clock.advance(10_000)
    const late = textOf(await $.ui.render(BAND))
    expect(late.length).toBeGreaterThan(early.length)
    expect(late).not.toContain('▍')
  })

  test('the prompt seeds a story with a thing the model knows', async ($, on) => {
    const clock = seat(on)
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work' })
    await clock.advance(20_000)
    await $.turn.start({ text: 'fix the flaky bug in the login page', turnId: 't1' })
    await clock.settle()
    const text = textOf(await $.ui.render(BAND))
    expect(text).toContain('little girl named')
    expect(text).toContain('found a bug.')
    expect(text).toContain('seeded by your prompt')
  })

  test('/story off hides the band and /story on restores it', async ($, on) => {
    const clock = seat(on)
    await $.session.start({ surface: 'terminal', isInteractive: true, cwd: '/work' })
    await clock.settle()
    const run = (args: string) =>
      $.command.run({
        command: 'story',
        args,
        origin: { kind: 'composer' },
        presentation: { isFullscreen: true, columns: 120 },
      })

    const off = await run('off')
    expect(off.text).toContain('hidden')
    expect(textOf(await $.ui.render(BAND))).toBe('')

    const on2 = await run('on')
    expect(on2.text).toContain('new story')
    await clock.settle()
    expect(textOf(await $.ui.render(BAND))).toContain('Once upon a time')

    const named = await run('zelda')
    expect(named.text).toContain('Zelda')
    await clock.settle()
    expect(textOf(await $.ui.render(BAND))).toContain('named Zelda.')
  })
})
