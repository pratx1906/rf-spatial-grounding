"""Cross-site AIR7311 comparison (band-matched array, full propagation, calibrated) -> markdown table + JSON."""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MATCHED = {"ism_2400": "B_2g4", "ism_5800": "A_5g8", "sub_ghz_868_915": "C_sub1g"}
LABEL = {"ism_2400": "2.4 GHz", "ism_5800": "5.8 GHz", "sub_ghz_868_915": "868/915 MHz"}


def main() -> None:
    rows, out = [], {}
    for s in ("S1", "S2", "S3", "S4", "S5"):
        f = REPO / "runs" / f"air7311_{s}" / "summary.json"
        if not f.exists():
            continue
        summ = json.loads(f.read_text())
        for band, arr in MATCHED.items():
            c = summ["combos"][f"{band}__spec_diff__{arr}__ic0"]["by_height"]
            out[f"{s}/{band}"] = c
            cells = []
            for h in ("5m", "15m", "40m", "80m"):
                b = c.get(h, {})
                cells.append(f"{b.get('music_gt10_pct', float('nan')):.1f}% ({b.get('detected_pct', 0):.0f}%)")
            rows.append(f"| {s} | {LABEL[band]} | " + " | ".join(cells) + " |")
    md = ["| Site | Band | 5 m | 15 m | 40 m | 80 m |", "|---|---|---|---|---|---|"] + rows
    (REPO / "docs" / "RESULTS" / "air7311_sites_table.md").write_text("\n".join(md) + "\n")
    (REPO / "docs" / "RESULTS" / "air7311_sites.json").write_text(json.dumps(out, indent=1))
    print("\n".join(md))


if __name__ == "__main__":
    main()
