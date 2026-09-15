# Storytime

A 260K-parameter transformer running inside Claude Code as a mod. It sits
above the prompt and writes a short story about whatever Claude is working
on: the tool Claude just called picks the kind of hero, the file it touched
names the hero, and a word from your prompt is what the hero finds. Every
token is generated locally by the mod's own TypeScript; nothing goes to the
API.

```
Storytime · TinyStories-260K · local transformer · 0 API tokens
Once upon a time, there was a little pencil named Auth. Auth loved to draw
pictures of the sun and the moon. One day, Auth saw a big tree…▍
seeded by Edit auth-service.ts · 4300 tok/s · /story
```

The model is Andrej Karpathy's `stories260K` from `karpathy/tinyllamas`
(dim 64, 5 layers, 8 heads, 4 KV heads, vocab 512), the same checkpoint
[gbc-transformer](https://github.com/maddiedreese/gbc-transformer) runs on
a Game Boy Color at one token every 2 minutes 49 seconds. Here it runs at a
few thousand tokens a second and is throttled to two tokens per 80 ms so it
reads as typing.

## Install

Mods are built on function hooks, which are early access in Claude Code
2.1.272 and later, so start Claude Code with the flag:

```sh
CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude
```

Then, inside Claude Code, add this repository as a marketplace and install
the plugin:

```
/plugin marketplace add maddiedreese/cc-storytime
/plugin install storytime@cc-storytime
```

Or run it from a clone without installing:

```sh
git clone https://github.com/maddiedreese/cc-storytime
CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude --plugin-dir ./cc-storytime
```

## Use

Work as usual. The band reseeds on each prompt you send and on tool calls,
at most once every twelve seconds.

- `/story` writes a new story.
- `/story <name>` writes one about that hero.
- `/story off` hides the band (remembered across sessions); `/story on` brings it back.

Heroes by tool: Edit is a pencil, Write a pen, Read a book, Bash a robot,
Grep a dog, Glob a bird, WebFetch a fish, Agent a bear, your own prompt a
girl. A file named `auth-service.ts` makes a hero named Auth; `git status`
makes one named Git. Developer words in your prompt become story things: a
bug stays a bug, a test is a puzzle, a branch is a tree, a deploy is a boat.

## How it is built

| file | what it is |
| --- | --- |
| `hooks/register.ts` | The hooks module: `session.start` binds `$` and registers `/story`; `turn.start` and `tool.call` reseed; `ui.render` of `AbovePrompt` draws the band over whatever is beneath it; a `$.clock.every` ticks generation. |
| `hooks/model.ts` | The transformer: RMSNorm, RoPE, grouped-query attention with a KV cache, SwiGLU, tied embeddings. Float32 typed arrays; a port of `run.c`. |
| `hooks/tokenizer.ts` | The llama2.c BPE tokenizer with byte fallback. |
| `hooks/story.ts` | One story: prefill, then a token per step, stopping on EOS or a sentence end. |
| `hooks/seed.ts` | Activity to opening line. |
| `hooks/views.tsx` | The band's three rows. |
| `hooks/weights.ts` | Generated: the int8 row-quantized weights (276 KB) and the tokenizer, base64, since a hooks module has no file access of its own for binaries. |
| `tools/pack.py` | Writes `weights.ts` from `stories260K.bin` and `tok512.bin`. |
| `tools/verify.ts` | `node tools/verify.ts`: checks the port's greedy output against the C reference and times it. |

The hooks environment has no Node and no DOM, only web globals, so the whole
runtime is plain ES2023 on typed arrays. Nothing here uses WebAssembly.

### Regenerating the weights

```sh
curl -LO https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/stories260K.bin
curl -LO https://huggingface.co/karpathy/tinyllamas/resolve/main/stories260K/tok512.bin
python3 tools/pack.py stories260K.bin tok512.bin hooks/weights.ts
```

## Tests

```sh
CLAUDE_CODE_ENABLE_FUNCTION_HOOKS=1 claude plugin test .
```

`tsc -p tsconfig.json` typechecks against `.claude/types/claude-code.d.ts`,
which `/plugin-types` writes inside a session with function hooks enabled.

## Credits

- TinyStories-260K and its tokenizer: Andrej Karpathy, `karpathy/tinyllamas`.
- The idea of drawing above the prompt with function hooks: sezaakgun's cc-arcade.

## License

MIT
