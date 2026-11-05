---
name: python-batch
description: "Use for loops over many files or generations in wangp_python: classify or caption a folder of pictures, transcribe clips, make a variation of each image, generate a series and pick the best, find a person across videos, reorganize files, with progress and safe stopping."
---

# Batches with WanGP's tools

A loop belongs in one script: the results stay in Python, the user sees progress, and only the summary reaches the conversation.

```python
# Photos with people in them, as a contact sheet
from PIL import Image, ImageOps, ImageDraw
entries = [e for e in wangp.listdir("@documents/photos", recursive=True) if e["kind"] == "image"]
hits = {}
for i, e in enumerate(entries):
    wangp.progress(i, len(entries), e["name"])  # as the item starts: the bar shows from the first item
    r = wangp.inspect(e["path"], "Is at least one person clearly visible? Answer yes or no first.")
    if r.get("answer", "").lower().startswith("yes"):
        hits[e["path"]] = r["answer"]
    print(f"{i + 1}/{len(entries)} {e['name']}: {r.get('answer', r.get('error', ''))[:50]}")
wangp.progress(len(entries), len(entries))
print(f"{len(hits)} of {len(entries)} photos show people")
```

- Ask the visual model short questions with a fixed answer form (yes/no, a number, one word, MATCH/NOMATCH first) so the script can decide. Ask about what is pictured, not what is written: "Is a cat drawn or photographed in this image (not only named in its text)?" keeps posters and infographics that merely mention the word out of the results.
- Keep results in a dict keyed by path: a Stop keeps the variables, so the next run continues from what is done.
- Before a long loop, tell the user the expected time (an inspection takes seconds, a generation much longer).
- To find one person across many videos: one inspection per candidate with the reference image plus three or four frames (`media_inputs=[{"media_id": ref}, {"media_id": video, "time_s": t}, ...]`), not grids of small crops.

## Generations in a loop

```python
# Three seeds of one prompt, the visual model picks the best
results = []
for seed in (11, 22, 33):
    wangp.progress(len(results), 3, f"seed {seed}")
    job = wangp.generate({"template": "gen_image/Z Image Turbo", "prompt": "a red bicycle against a cafe wall, morning sun", "seed": seed})
    results.append(job)
wangp.progress(3, 3)
results[-1]
```

- Pass a template name plus the few overrides that change per item (prompt, seed, resolution, `activated_loras`); keep the seed fixed when the items should match.
- Read the returned job result for the new media ids (print its keys once), then compare them with one `wangp.inspect` call holding all candidates (`media_ids=[...]`).
- A list of settings in one `wangp.generate([...])` queues them together.

## Files

`wangp.io("mkdir", {"path": ...})`, `wangp.io("move", {"source": ..., "destination": ...})`, `copy` and `zip` reorganize authorized folders from the loop (deleting is a separate `wangp_io` call outside scripts). New files are made in the sandbox, then copied there.
