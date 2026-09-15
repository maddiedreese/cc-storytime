/* @jsxRuntime classic */
/* @jsx h */
/* @jsxFrag Fragment */
import type { Elements, RenderElement } from 'claude-code'

export type BandKit = Pick<Elements['terminal'], 'Box' | 'Text'>

export type BandModel = {
  readonly text: string
  readonly done: boolean
  readonly because: string
  readonly tokensPerSecond: number
  readonly maxRows: number
  readonly columns: number
}

export const TITLE = 'Storytime · TinyStories-260K · local transformer · 0 API tokens'

/**
 * The band's three rows: a dim title, the story (wrapped, a cursor while it
 * types), and a dim footer naming what seeded it. Under three rows only the
 * story shows.
 */
export function bandView(kit: BandKit, model: BandModel): RenderElement {
  const { Box, Text } = kit
  const story = model.done ? model.text : `${model.text}▍`
  const rate = model.tokensPerSecond > 0 ? ` · ${Math.round(model.tokensPerSecond)} tok/s` : ''
  const footer = `${model.because}${rate} · /story`
  const isTall = model.maxRows >= 3

  return (
    <Box flexDirection="column" width={model.columns} paddingX={1}>
      {isTall ? <Text dimColor>{TITLE}</Text> : null}
      <Text wrap="wrap">{story}</Text>
      {isTall ? <Text dimColor>{footer}</Text> : null}
    </Box>
  )
}
