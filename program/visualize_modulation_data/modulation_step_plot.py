"""
Generate an interactive HTML (Plotly.js) step graph for `modulation_index`.

- Reads `data/analyzed/*.json`
- Builds a dropdown to select ONE song
- Renders a step graph (stairs) of modulation_index over section order

Usage:
  python -m program.visualize_modulation_data.modulation_step_plot \
    --data_dir data/analyzed \
    --out_html vis_system_modulation/modulation_index.html

Tip:
  For best performance, this script writes a separate data file next to the HTML:
    - modulation_index_data.json
  Open the HTML via a local server (recommended):
    cd vis_system_modulation && python -m http.server 8000
    then open http://localhost:8000/modulation_index.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from program.analyze_data.chord_normalization import normalize_key_label

EMOTION_COLORS = {
    "JOY": "#FFFF73",
    "ANTICIPATION": "#F3AB63",
    "TRUST": "#88FC6E",
    "SADNESS": "#5150F8",
    "SURPRISE": "#74BBF9",
    "ANGER": "#E93323",
    "FEAR": "#429429",
    "DISGUST": "#EB60F8",
}


def _emotion_to_color(emotion_dict: Any) -> str:
    if not isinstance(emotion_dict, dict) or not emotion_dict:
        return "rgba(0,0,0,0)"
    try:
        emo, val = max(emotion_dict.items(), key=lambda kv: float(kv[1]))
        val_f = float(val)
    except Exception:
        return "rgba(0,0,0,0)"

    # If the strongest emotion is weak, treat as "no color"
    if val_f < 0.5:
        return "rgba(0,0,0,0)"

    return EMOTION_COLORS.get(str(emo), "rgba(0,0,0,0)")


def _load_song_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    return None


def _song_label(song: Dict[str, Any]) -> str:
    # For dropdown/title: no spotify_id, but include title + credits
    title = song.get("title") or "Unknown"
    artist = song.get("artist") or "Unknown"
    lyricist = song.get("lyricist") or "Unknown"
    composer = song.get("composer") or "Unknown"
    return f"{title} (アーティスト: {artist}, 作詞: {lyricist}, 作曲: {composer})"


def _build_series(song: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    secs = song.get("analyzed_chord_progressions_and_lyrics") or []
    if not isinstance(secs, list) or not secs:
        return None

    # If modulation_index is missing in analyzed JSON, compute it from section keys on the fly.
    tonic_key = None
    for sec in secs:
        if not isinstance(sec, dict):
            continue
        k = normalize_key_label(sec.get("key")) if sec.get("key") else None
        if k:
            tonic_key = k
            break

    prev_key_norm = None
    cur_mod_index = 0

    x: List[int] = []
    key_y: List[str] = []
    hover: List[str] = []
    confs: List[float] = []
    lyrics_short: List[str] = []
    lyric_bg: List[str] = []
    has_modulation = False

    chord_cursor = 0
    for i, sec in enumerate(secs):
        if not isinstance(sec, dict):
            continue
        mi = sec.get("modulation_index", None)
        if mi is None:
            key_norm = normalize_key_label(sec.get("key")) if sec.get("key") else None
            if tonic_key and key_norm:
                if key_norm == tonic_key:
                    cur_mod_index = 0
                else:
                    if prev_key_norm is None:
                        cur_mod_index = 1
                    elif key_norm != prev_key_norm:
                        if prev_key_norm == tonic_key:
                            cur_mod_index = 1
                        elif key_norm == tonic_key:
                            cur_mod_index = 0
                        else:
                            cur_mod_index += 1
                prev_key_norm = key_norm
                mi = cur_mod_index
            else:
                chord_cursor += len(sec.get("chord_progression") or [])
                continue

        key = sec.get("key", "")
        key_norm = normalize_key_label(key) if key else None
        conf = sec.get("key_confidence", None)
        try:
            conf_f = float(conf) if conf is not None else None
        except Exception:
            conf_f = None

        lyric = (sec.get("lyric") or "").replace("\n", " ").strip()
        if len(lyric) > 60:
            lyric = lyric[:60] + "…"

        # x-axis uses section index (uniform spacing)
        x.append(int(i))
        key_y.append(key_norm or key or "N/A")
        lyrics_short.append(lyric)
        lyric_bg.append(_emotion_to_color(sec.get("emotion")))
        # track if this song ever modulates (index >= 1)
        try:
            if int(mi) >= 1:
                has_modulation = True
        except Exception:
            pass

        if conf_f is None:
            # lyric is shown on x-axis labels; keep hover concise
            hover.append(f"sec={i}<br>key={key}<br>key_confidence=N/A<br>modulation_index={mi}")
            confs.append(1.0)
        else:
            hover.append(f"sec={i}<br>key={key}<br>key_confidence={conf_f:.3f}<br>modulation_index={mi}")
            confs.append(conf_f)

        chord_cursor += len(sec.get("chord_progression") or [])

    if not x:
        return None

    # X-axis labels: show ALL sections. Do not insert line breaks.
    def _label(s: str) -> str:
        return (s or "").strip()

    return {
        "id": song.get("spotify_id") or song.get("jtotal_path") or _song_label(song),
        "label": _song_label(song),
        "has_modulation": has_modulation,
        "x": x,
        "key_y": key_y,
        "x_label_vals": x,
        "x_label_text": [_label(s) for s in lyrics_short],
        "x_label_bg": lyric_bg,
        "hover": hover,
        "conf": confs,
    }


HTML_TEMPLATE = """<!doctype html>
<html lang="ja">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>modulation_index step graph</title>
  <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Noto Sans JP", "Hiragino Kaku Gothic ProN", Meiryo, sans-serif; margin: 16px; }}
    .row {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
    select {{ font-size: 14px; padding: 6px 8px; min-width: 420px; }}
    input[type="text"] {{ font-size: 14px; padding: 6px 8px; min-width: 320px; }}
    #chart {{ width: 100%; height: 78vh; }}
    .note {{ color: #555; font-size: 12px; }}
  </style>
</head>
<body>
  <div class="row">
    <div><b>曲を選択:</b></div>
    <input id="filterInput" type="text" placeholder="フィルタ（例: アーティスト名 / 曲名）" />
    <select id="songSelect"></select>
    <div class="note">x: セクションindex / y: key（キー推定）</div>
  </div>
  <div id="chart"></div>

  <script>
    const DATA_URL = {data_url_json};
    let SONGS = [];

    const select = document.getElementById("songSelect");
    const filterInput = document.getElementById("filterInput");

    function _labelForSong(s) {{
      return (s.has_modulation ? "★" : "") + s.label;
    }}

    function populate(filterText = "") {{
      select.innerHTML = "";
      // placeholder (no selection)
      const ph = document.createElement("option");
      ph.value = "";
      ph.textContent = "-- 選択 --";
      select.appendChild(ph);

      const ft = (filterText || "").toLowerCase().trim();
      for (const s of SONGS) {{
        const display = _labelForSong(s);
        if (ft && !display.toLowerCase().includes(ft)) {{
          continue;
        }}
        const opt = document.createElement("option");
        opt.value = s.id;
        opt.textContent = display;
        select.appendChild(opt);
      }}
    }}

    function renderSong(songId) {{
      if (!songId) {{
        Plotly.purge("chart");
        document.getElementById("chart").textContent = "";
        return;
      }}
      const s = SONGS.find(x => x.id === songId) || SONGS[0];

      // Build step trace arrays, but BREAK the blue line where we draw transition overlay.
      // This hides the blue segment on intervals that are flagged as transition.
      const xStep = [];
      const yStep = [];
      for (let i = 0; i < s.x.length; i++) {{
        xStep.push(s.x[i]);
        yStep.push(s.key_y[i]);
        if (i < s.x.length - 1) {{
          const c0 = (s.conf && s.conf[i] != null) ? s.conf[i] : 1.0;
          const c1 = (s.conf && s.conf[i+1] != null) ? s.conf[i+1] : 1.0;
          if (c0 < 0.999 || c1 < 0.999) {{
            xStep.push(null);
            yStep.push(null);
          }}
        }}
      }}

      const traceStep = {{
        x: xStep,
        y: yStep,
        text: s.hover,
        hoverinfo: "text",
        mode: "lines+markers",
        line: {{ shape: "hv", width: 2 }},
        marker: {{
          size: 7,
          color: (s.conf || []).map(c => (c < 0.999 ? "rgba(255,140,0,0.95)" : "rgba(31,119,180,0.95)")),
          line: {{ width: 0 }}
        }},
        name: "key (estimated)"
      }};

      // Overlay: draw diagonal/dotted segments where confidence is not 1.0
      const xLin = [];
      const yLin = [];
      for (let i = 0; i < s.x.length - 1; i++) {{
        const c0 = (s.conf && s.conf[i] != null) ? s.conf[i] : 1.0;
        const c1 = (s.conf && s.conf[i+1] != null) ? s.conf[i+1] : 1.0;
        if (c0 < 0.999 || c1 < 0.999) {{
          xLin.push(s.x[i], s.x[i+1], null);
          yLin.push(s.key_y[i], s.key_y[i+1], null);
        }}
      }}
      const traceTransition = {{
        x: xLin,
        y: yLin,
        mode: "lines",
        hoverinfo: "skip",
        line: {{ shape: "linear", width: 3, color: "rgba(255,140,0,0.9)" }},
        name: "transition (confidence<1)"
      }};

      // Draw lyrics as annotations with emotion-colored background.
      const ann = [];
      const vals = s.x_label_vals || [];
      const texts = s.x_label_text || [];
      const bgs = s.x_label_bg || [];
      for (let i = 0; i < vals.length; i++) {{
        ann.push({{
          x: vals[i],
          y: -0.25,
          xref: "x",
          yref: "paper",
          text: texts[i] || "",
          textangle: 90,
          showarrow: false,
          // Center-align labels to the same x as markers to avoid visual "shift".
          xanchor: "center",
          yanchor: "top",
          align: "center",
          bgcolor: bgs[i] || "rgba(255,255,255,0.9)",
          opacity: 0.85,
          bordercolor: "rgba(0,0,0,0.08)",
          borderwidth: 1,
          font: {{ size: 10, color: "#111" }}
        }});
      }}

      const layout = {{
        title: {{ text: (s.has_modulation ? "★" : "") + s.label, x: 0.02, xanchor: "left" }},
        xaxis: {{
          title: "section index",
          zeroline: false,
          showticklabels: false
        }},
        yaxis: {{ title: "key (estimated)" }},
        margin: {{ l: 60, r: 20, t: 60, b: 320 }},
        annotations: ann,
        hovermode: "closest"
      }};

      Plotly.react("chart", [traceStep, traceTransition], layout, {{displayModeBar: true}});
    }}

    select.addEventListener("change", (e) => {{
      const id = e.target.value;
      // Update URL so it can be shared/bookmarked.
      const params = new URLSearchParams(window.location.search);
      if (id) {{
        params.set("spotify_id", id);
      }} else {{
        params.delete("spotify_id");
      }}
      const newUrl = window.location.pathname + (params.toString() ? ("?" + params.toString()) : "");
      window.history.replaceState(null, "", newUrl);
      renderSong(id);
    }});
    filterInput.addEventListener("input", (e) => {{
      const prev = select.value;
      populate(e.target.value);
      // keep selection if still present; otherwise render first option
      if (prev && Array.from(select.options).some(o => o.value === prev)) {{
        select.value = prev;
        renderSong(prev);
      }} else {{
        select.value = "";
        renderSong("");
      }}
    }});

    async function boot() {{
      try {{
        const res = await fetch(DATA_URL, {{cache: "no-store"}});
        SONGS = await res.json();
        if (!Array.isArray(SONGS) || SONGS.length === 0) {{
          throw new Error("No songs in data");
        }}
        populate("");
        // If URL has ?spotify_id=..., auto-select and render.
        const params = new URLSearchParams(window.location.search);
        const initialId = params.get("spotify_id") || "";
        if (initialId && Array.from(select.options).some(o => o.value === initialId)) {{
          select.value = initialId;
          renderSong(initialId);
        }} else {{
          // Do not render until user selects a song
          select.value = "";
        }}
      }} catch (e) {{
        document.getElementById("chart").textContent =
          "Failed to load data (" + String(e) + ").\\n" +
          "If you opened this HTML as file://, use a local server instead.\\n" +
          "Example: cd vis_system_modulation && python -m http.server 8000";
      }}
    }}
    boot();
  </script>
</body>
</html>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/analyzed", help="Directory containing analyzed *.json files")
    ap.add_argument("--out_html", default="vis_system_modulation/modulation_index.html", help="Output HTML path")
    ap.add_argument("--out_data", default=None, help="Output data JSON path (default: next to out_html)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    paths = sorted(data_dir.glob("*.json"))

    songs: List[Dict[str, Any]] = []
    for p in paths:
        song = _load_song_json(p)
        if not song:
            continue
        series = _build_series(song)
        if series:
            songs.append(series)

    songs.sort(key=lambda s: s["label"])

    out_path = Path(args.out_html)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_data = Path(args.out_data) if args.out_data else (out_path.parent / "modulation_index_data.json")
    out_data.write_text(json.dumps(songs, ensure_ascii=False), encoding="utf-8")

    data_url = out_data.name
    html = HTML_TEMPLATE.format(data_url_json=json.dumps(data_url, ensure_ascii=False))
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote: {out_path}")
    print(f"Wrote: {out_data} (songs={len(songs)})")


if __name__ == "__main__":
    main()

