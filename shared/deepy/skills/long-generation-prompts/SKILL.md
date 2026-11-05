---
name: long-generation-prompts
description: Author and revise generation prompts longer than 4096 characters or tokens, including WanGP sliding-window prompts whose windows must remain consistent and correctly separated.
---

# Long Generation Prompts

Keep the authoritative prompt in `@documents/prompt.txt` and, when useful, a compact plan in `@documents/prompt_plan.md`. Use `rg` for targeted inspection, `append_text` for bounded additions, and `edit` for exact replacement; do not reconstruct the complete prompt from conversational memory.

For a WanGP sliding-window prompt, the entire file is one prompt. Separate complete windows with exactly one empty line. Inside a window, use single newlines between labels or clauses and never use an empty line. Preserve window order.

Before writing, define the invariant elements shared by all windows: subject identity, appearance, wardrobe, environment, visual or audio style, camera grammar, lighting, palette, and prohibited discontinuities. Define each window's intended change separately. Repeat identity-critical wording where the generation model benefits from explicit reinforcement.

Use this workflow:

1. Create the plan and prompt with `append_text(file_path, text)`, which creates a missing file. Text is literal: include every newline and do not prefix lines with `+`.
2. Keep each window's unique, searchable title as a full-line comment: `# WINDOW 03 — ...`. WanGP removes these comments before splitting windows and generating, while `rg` can still find them in the source file. Put duration/overlap commands and model prompt content on the following lines; an inline `#` after a command or prompt is not a comment.
3. Search neighboring window comment titles and invariant phrases before revising a window.
4. Revise only the target window with `edit(file_path, old_string, new_string)`. Copy an exact, unique old string, including adjacent comment titles when useful; whitespace and line endings must match.
5. Search `^# WINDOW` to confirm order and count, then search `^$` to inspect blank-line boundaries.
6. Check that blank-line-separated window count matches any per-window images or other ordered conditioning inputs.

When extending the prompt, append one or a few complete windows at a time. Start appended text with exactly two newline characters when the existing final window has no trailing separator. `append_text` adds nothing implicitly. Use `replace_all=true` only for a deliberately verified global invariant change.

For literal searches, use a call such as `-n -F "# WINDOW 03" -C 12 -- @documents/prompt.txt`. For boundary checks, use `-n "^$" -- @documents/prompt.txt`.

Example file layout (replace the prompt text with the model's required structure):

```text
# WINDOW 01 — NEON STREET
[/duration=12s]
First window's model-specific prompt content.

# WINDOW 02 — ROOFTOP
[/duration=12s]
Second window's model-specific prompt content.
```

To use the completed file in image, video, or audio generation, set the relevant prompt field to the exact standalone value `@file("@documents/prompt.txt")`. WanGP reads and snapshots the UTF-8 content before queueing, so the queued task does not depend on the transient file afterward. The marker must occupy the whole prompt field; do not combine it with inline text.
