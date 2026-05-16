"""Regenerate docs/index.html with all Python sources inlined as base64.

Why inline? The stlite `mount({files: {... : {url}}})` form requires the
HTML to be served alongside the .py files. Inlining removes that
constraint — the resulting index.html is self-sufficient and works under
file://, any static server, GitHub Pages, etc.

Run from the repo root:
    python docs/build_index.py
"""

from __future__ import annotations

import base64
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

# (virtual filesystem path, source-on-disk path)
FILES = [
    ("streamlit_app.py",          HERE / "streamlit_app.py"),
    ("phare_load/__init__.py",    HERE / "phare_load" / "__init__.py"),
    ("phare_load/config.py",      HERE / "phare_load" / "config.py"),
    ("phare_load/geometry.py",    HERE / "phare_load" / "geometry.py"),
    ("phare_load/load.py",        HERE / "phare_load" / "load.py"),
    ("phare_load/models.py",      HERE / "phare_load" / "models.py"),
    ("phare_load/plotting.py",    HERE / "phare_load" / "plotting.py"),
    ("phare_load/plotting2d.py",  HERE / "phare_load" / "plotting2d.py"),
    ("phare_load/plotting3d.py",  HERE / "phare_load" / "plotting3d.py"),
]


def main() -> None:
    files_b64 = {
        vpath: base64.b64encode(p.read_bytes()).decode("ascii")
        for vpath, p in FILES
    }
    payload_json = json.dumps(files_b64, indent=2)

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PHARE load estimator — interactive</title>
  <meta name="description"
        content="Interactive estimator for compute/memory load of PHARE global magnetosphere PIC runs." />
  <link rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/@stlite/browser@0.85.1/build/stlite.css" />
  <style>
    html, body {{ margin: 0; padding: 0; height: 100%; font-family: system-ui, sans-serif; }}
    #root {{ height: 100%; }}
    .loading {{ padding: 2rem; max-width: 720px; margin: 4rem auto; color: #333; }}
    .loading h1 {{ margin-top: 0; font-weight: 600; }}
    .loading .hint {{ color: #666; font-size: 0.95rem; line-height: 1.5; }}
  </style>
</head>
<body>
  <div id="root">
    <div class="loading">
      <h1>PHARE load estimator</h1>
      <p class="hint">
        Loading Python runtime in your browser
        (≈25–35 MB on first visit, cached afterwards).
        This page runs entirely client-side via
        <a href="https://stlite.net" target="_blank" rel="noopener">stlite</a> +
        <a href="https://pyodide.org" target="_blank" rel="noopener">Pyodide</a>.
        No data leaves your machine.
      </p>
      <p class="hint">
        Source: <a href="https://github.com/PHAREHUB/pharculator">github.com/PHAREHUB/pharculator</a> ·
        Model: <a href="https://github.com/PHAREHUB/pharculator/blob/main/MODEL.md">MODEL.md</a>
      </p>
    </div>
  </div>

  <script type="application/json" id="phare-load-files">
{payload_json}
  </script>

  <script type="module">
    import {{ mount }} from "https://cdn.jsdelivr.net/npm/@stlite/browser@0.85.1/build/stlite.js";

    function b64ToText(b64) {{
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return new TextDecoder("utf-8").decode(bytes);
    }}

    const raw = JSON.parse(document.getElementById("phare-load-files").textContent);
    const files = {{}};
    for (const [path, b64] of Object.entries(raw)) {{
      files[path] = b64ToText(b64);
    }}

    mount(
      {{
        requirements: [
          "numpy",
          "matplotlib",
          "plotly",
          "scikit-image",
        ],
        entrypoint: "streamlit_app.py",
        files: files,
      }},
      document.getElementById("root")
    );
  </script>
</body>
</html>
"""

    out = HERE / "index.html"
    out.write_text(html, encoding="utf-8")
    total_bytes = sum(len(b) for b in files_b64.values())
    print(f"Wrote {out.relative_to(ROOT)} "
          f"({len(html):,} bytes, payload {total_bytes:,} b64 bytes "
          f"covering {len(files_b64)} files)")


if __name__ == "__main__":
    main()
