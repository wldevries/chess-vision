"""Build a standalone HTML gallery of the blurriest store images for human review.

Reads `data/_blur_scores.json` (variance-of-Laplacian focus measure per piece-labelled store
image; lower = blurrier), shows everything below `--thresh` sorted blurriest-first so a human
can eyeball where blur actually starts. The decided cutoff/exceptions then feed a blur-exclude
list so the blur augmentation skips already-soft frames.

    uv run python scripts/build_blur_review.py --thresh 450
    # open data/_blur_review.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scores", type=Path, default=Path("data/_blur_scores.json"))
    p.add_argument("--out", type=Path, default=Path("data/_blur_review.html"))
    p.add_argument("--top", type=int, default=120, help="show the N blurriest by Crete metric (cb)")
    args = p.parse_args()

    rows = json.loads(args.scores.read_text(encoding="utf-8"))
    rows.sort(key=lambda r: r["cb"], reverse=True)  # blurriest (highest cb) first
    cand = rows[: args.top]

    def card(r: dict) -> str:
        src = "store/" + r["image"]  # out lives in data/, images at data/store/<relpath>
        cb = r["cb"]
        hue = 0 if cb > 0.62 else (35 if cb > 0.55 else 55)  # red->amber->yellow by blurriness
        return (
            f'<figure><a href="{html.escape(src)}" target="_blank">'
            f'<img loading="lazy" src="{html.escape(src)}"></a>'
            f'<figcaption><span class="fm" style="background:hsl({hue} 80% 45%)">{cb:.2f}</span>'
            f"{html.escape(r['board'])}<br><small>{html.escape(r['image'])}</small>"
            f"</figcaption></figure>"
        )

    cards = "\n".join(card(r) for r in cand)
    doc = f"""<!doctype html><meta charset=utf-8><title>Blur review</title>
<style>
 body{{font:14px system-ui;margin:16px;background:#111;color:#eee}}
 h1{{font-size:18px}} .note{{color:#aaa;max-width:72ch;line-height:1.5}}
 .grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:12px;margin-top:16px}}
 figure{{margin:0;background:#1c1c1c;border-radius:8px;overflow:hidden}}
 img{{width:100%;height:200px;object-fit:cover;display:block;background:#000}}
 figcaption{{padding:6px 8px;font-size:12px;color:#ccc}}
 .fm{{color:#000;font-weight:700;border-radius:4px;padding:1px 6px;margin-right:6px}}
 small{{color:#888;word-break:break-all}}
</style>
<h1>Blur review &mdash; {len(cand)} blurriest images (Crete perceptual metric)</h1>
<p class=note>Sorted blurriest&rarr;sharpest by the Crete re-blur metric (badge = cb, higher =
blurrier; red &gt;0.62). It's content-normalized so it tracks perceived blur better than the old
focus measure, but it's not perfect. Scroll until the images look acceptably sharp and note the
badge there. Click any image for full-size. Then tell me the cutoff (e.g. &ldquo;above 0.60 is
blurry&rdquo;) or list exceptions, and I'll mark those so the blur augmentation skips them.</p>
<div class=grid>
{cards}
</div>"""
    args.out.write_text(doc, encoding="utf-8")
    print(f"wrote {args.out} with the {len(cand)} blurriest images (by Crete cb)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
