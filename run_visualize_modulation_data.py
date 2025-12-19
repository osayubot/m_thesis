#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
modulation_index 可視化（Plotly HTML生成）エントリーポイント

使用方法:
  python run_visualize_modulation_data.py [data_dir] [out_html]

例:
  python run_visualize_modulation_data.py data/analyzed vis_system/modulation/modulation_index.html
"""

import sys
from pathlib import Path

from program.visualize_modulation_data.modulation_step_plot import main as modulation_main


if __name__ == "__main__":
    script_dir = Path(__file__).parent
    default_data_dir = str(script_dir / "data" / "analyzed")
    default_out_html = str(script_dir / "vis_system" / "modulation" / "modulation_index.html")

    data_dir = sys.argv[1] if len(sys.argv) > 1 else default_data_dir
    out_html = sys.argv[2] if len(sys.argv) > 2 else default_out_html

    print("=" * 60)
    print("転調（modulation_index）可視化HTML生成")
    print("=" * 60)
    print(f"データディレクトリ: {data_dir}")
    print(f"出力HTML: {out_html}")
    print("=" * 60)

    # primary output (default: vis_system/modulation)
    sys.argv = ["modulation_step_plot", "--data_dir", data_dir, "--out_html", out_html]
    modulation_main()


