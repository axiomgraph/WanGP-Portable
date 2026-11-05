---
name: python-data
description: "Use for data work in wangp_python: tables and CSV files, statistics, charts and infographics, reports in Markdown, figures read from text files or PDFs, and media metadata summaries."
---

# Data, charts and reports

## Sources

- Text and CSV files: `open("@documents/data.csv")` or `pandas.read_csv(wangp.path(ref))`.
- PDFs: `from pypdf import PdfReader; PdfReader(wangp.path(ref)).pages[i].extract_text()` (text layers only; scanned pages are images: read them with `inspect_media`).
- Web data: `wangp.download(url)` saves a page or file under `@documents/web` (its `path` opens with `open()`), `wangp.web('open', url=...)` returns a page's text; a page read with `wangp_web_search` outside the script works too. Raw network modules are not available.
- Media metadata: `wangp.io("info", {"paths": [...]})` (20 per call) or `wangp.listdir` for sizes and dates.
- Many audio files: measure them in one script, not one tool call each. `soundfile` gives the samples (duration, peak and RMS in dBFS with numpy); for loudness (LUFS), silences or pitch, loop `wangp.toolbox("inspect_audio", media_id=e["path"])` over `wangp.listdir(folder, recursive=True)` and keep the fields you need.

## Charts

```python
# Monthly sales chart
import matplotlib.pyplot as plt
W, H, dpi = 1280, 720, 100
fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="#101418")
ax.set_facecolor("#101418")
months, values = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"], [12, 18, 9, 25, 31, 27]
bars = ax.bar(months, values, color="#4cc9f0", width=0.6)
ax.bar_label(bars, color="white", fontsize=16, padding=4)
ax.set_title("Sales 2026 (k units)", color="white", fontsize=26, pad=20)
ax.tick_params(colors="white", labelsize=16)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.set_yticks([])
fig.savefig("out/sales.png", dpi=dpi, facecolor=fig.get_facecolor())
```

Size the figure in pixels (`figsize = W / dpi, H / dpi`), label values directly instead of relying on axes, one accent colour, and check the image with `inspect_media`. An infographic is a chart plus text panels composed with PIL (see `wangp://skills/python-images`).

A 3D chart (bars, columns, a city of values) is a mesh, not matplotlib's 3D axes: boxes sized to the values with their name and figure as `wangp.text_mesh` labels on their faces, then `render_3d_views` with `turntable_s` for the video (see `wangp://skills/python-3d`). The labels stay on their bars at every angle and the chart fills the frame; matplotlib's 3D text overlaps and its axes need units.

## Outputs

- CSV / Markdown deliverables: write them in the sandbox (`out/` publishes only media), then `wangp.io("copy", {"source": "@documents/sandbox/scratch/report.csv", "destination": "@documents/report.csv"})`, or return the text and save it with `wangp_io write_text`.
- Print the table you will report (a few rows), not the raw data.
