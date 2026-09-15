/**
 * From what Claude is doing to the opening of a story: the tool names the
 * kind of hero, the file it touched names the hero, and a common word from
 * the person's prompt is what the hero finds. TinyStories knows few words,
 * so the file stem goes in the name slot (any token works as a name) and
 * the theme is kept only when it is one whole piece of the vocabulary.
 */

export type Activity = {
  /** The tool Claude called, or `prompt` for the person's own words. */
  readonly tool: string
  /** A path the call named, if any. */
  readonly path?: string
  /** A shell command's text, if any. */
  readonly command?: string
  /** The person's prompt text, when `tool` is `prompt`. */
  readonly text?: string
}

export type Seed = {
  /** The text the model continues from. */
  readonly prompt: string
  /** A short line saying what seeded it, for the band's footer. */
  readonly because: string
  /** The hero's name, kept so the sampler's seed varies per hero. */
  readonly name: string
}

const HERO_OF: Readonly<Record<string, string>> = {
  Edit: 'pencil',
  Write: 'pen',
  NotebookEdit: 'notebook',
  Read: 'book',
  Glob: 'bird',
  Grep: 'dog',
  Bash: 'robot',
  PowerShell: 'robot',
  WebFetch: 'fish',
  WebSearch: 'owl',
  Agent: 'bear',
  Skill: 'wizard',
  AskUserQuestion: 'cat',
  prompt: 'girl',
}

const DEFAULT_HERO = 'mouse'

/**
 * Things TinyStories has seen many times, so the model can carry one into
 * a story. The 512-piece vocabulary has almost no whole words in it, so
 * "the model knows it" is a list, not a tokenizer check.
 */
const KNOWN_THINGS = new Set([
  'ball', 'tree', 'cat', 'dog', 'park', 'flower', 'cake', 'toy', 'box', 'hat',
  'car', 'sun', 'star', 'bird', 'fish', 'book', 'bed', 'door', 'house', 'apple',
  'frog', 'duck', 'bear', 'bunny', 'kite', 'boat', 'train', 'truck', 'cookie',
  'candy', 'garden', 'beach', 'river', 'lake', 'hill', 'rock', 'stick', 'leaf',
  'egg', 'nest', 'key', 'coin', 'ring', 'shell', 'cloud', 'rain', 'snow', 'moon',
  'bug', 'ant', 'bee', 'butterfly', 'puppy', 'kitten', 'pig', 'cow', 'horse',
  'lion', 'monkey', 'mouse', 'owl', 'robot', 'dragon', 'castle', 'tower',
  'bridge', 'road', 'map', 'letter', 'picture', 'song', 'game', 'puzzle', 'gift',
  'balloon', 'bubble', 'button', 'cup', 'spoon', 'blanket', 'pillow', 'window',
  'ladder', 'bucket', 'basket', 'log', 'lock', 'clock', 'bell', 'drum', 'flag',
])

/** What a developer's words are in a story: a bug is a bug, a test a puzzle. */
const THING_OF_JARGON: Readonly<Record<string, string>> = {
  bug: 'bug', bugs: 'bug', error: 'bug', errors: 'bug', crash: 'bug',
  test: 'puzzle', tests: 'puzzle', testing: 'puzzle',
  file: 'book', files: 'book', doc: 'book', docs: 'book', readme: 'book',
  build: 'tower', compile: 'tower', deploy: 'boat', release: 'boat', ship: 'boat',
  branch: 'tree', merge: 'bridge', commit: 'gift', push: 'kite', pull: 'rope',
  cache: 'box', config: 'map', log: 'log', logs: 'log', token: 'coin', key: 'key',
  server: 'castle', database: 'basket', model: 'robot', api: 'bell', script: 'song',
  function: 'button', loop: 'ring', string: 'string', array: 'basket', class: 'box',
  password: 'lock', secret: 'lock', network: 'bridge', cloud: 'cloud',
  story: 'book', game: 'game', music: 'song', picture: 'picture', image: 'picture',
}

/** The file's stem as a name: `auth-service.ts` becomes `Auth`. */
export function nameOfPath(path: string): string | null {
  const base = path.split(/[\\/]/).pop() ?? ''
  const stem = base.replace(/\.[^.]*$/, '')
  const word = stem.split(/[^A-Za-z]+/).find(part => part.length >= 2)
  return word ? word[0]!.toUpperCase() + word.slice(1).toLowerCase() : null
}

/** The command's program, capitalized, as a name: `git status` becomes `Git`. */
export function nameOfCommand(command: string): string | null {
  const first = command.trim().split(/\s+/)[0] ?? ''
  const word = first.split(/[\\/]/).pop()?.replace(/[^A-Za-z].*$/, '') ?? ''
  return word.length >= 2 ? word[0]!.toUpperCase() + word.slice(1).toLowerCase() : null
}

/**
 * A thing from the prompt for the hero to find: a developer's word mapped
 * to its story counterpart first, else the first thing the model knows.
 */
export function themeOf(text: string): string | null {
  const words = text.toLowerCase().match(/[a-z]+/g) ?? []
  for (const word of words) {
    const mapped = THING_OF_JARGON[word]
    if (mapped) return mapped
  }
  for (const word of words) {
    const singular = word.endsWith('s') ? word.slice(0, -1) : word
    if (KNOWN_THINGS.has(word)) return word
    if (KNOWN_THINGS.has(singular)) return singular
  }
  return null
}

/** Names a hero for a story that nothing seeded. */
export const IDLE_NAMES = ['Lily', 'Ben', 'Mia', 'Tom', 'Sue', 'Max'] as const

export function seedOf(activity: Activity | null, fallbackName: string = IDLE_NAMES[0]): Seed {
  const hero = HERO_OF[activity?.tool ?? ''] ?? DEFAULT_HERO
  const name =
    (activity?.path && nameOfPath(activity.path)) ||
    (activity?.command && nameOfCommand(activity.command)) ||
    fallbackName
  const theme = activity?.text ? themeOf(activity.text) : null
  const opening = `Once upon a time, there was a little ${hero} named ${name}.`
  const prompt = theme ? `${opening} One day, ${name} found a ${theme}.` : opening
  const because = !activity
    ? 'waiting for Claude'
    : activity.tool === 'prompt'
      ? 'your prompt'
      : activity.path
        ? `${activity.tool} ${activity.path.split(/[\\/]/).pop()}`
        : activity.command
          ? `${activity.tool} ${activity.command.trim().split(/\s+/)[0]}`
          : activity.tool
  return { prompt, because, name }
}
