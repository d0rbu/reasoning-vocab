"""
Visualize reasoning token usage in generated outputs.

This script will:
- Generate outputs from model checkpoints
- Highlight reasoning tokens with unique colors
- Create three output formats: image, markdown, and LaTeX
- Track token multiplicity for reasoning tokens
"""

# TODO: Implement reasoning output visualization
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

try:  # Optional dependency
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:  # pragma: no cover - environment without matplotlib
    HAS_MPL = False


@dataclass(frozen=True)
class RenderOutputs:
    """Container for the three render outputs.

    Attributes:
        image_path: Path to saved PNG image (None if image rendering disabled/unavailable)
        markdown_text: Markdown-formatted string
        latex_text: LaTeX document string
    """

    image_path: Path | None
    markdown_text: str
    latex_text: str


# A small, visually distinct color palette (hex)
# Note: Kept short intentionally; hashing selects deterministically
DEFAULT_PALETTE: tuple[str, ...] = (
    "#ff6b6b",  # red
    "#4ecdc4",  # teal
    "#ffd93d",  # yellow
    "#5c7cfa",  # indigo
    "#f06595",  # pink
    "#51cf66",  # green
    "#845ef7",  # violet
    "#ffa94d",  # orange
)


def _stable_index_for_token(token: str, multiplicity: int, palette_size: int) -> int:
    """Return a deterministic palette index for a (token, multiplicity) pair.

    Uses BLAKE2b hash truncated to 8 bytes for speed and stability.
    """
    h = hashlib.blake2b(f"{token}::{multiplicity}".encode(), digest_size=8).digest()
    as_int = int.from_bytes(h, byteorder="big", signed=False)
    return as_int % max(1, palette_size)


def color_for_token(token: str, multiplicity: int, palette: Sequence[str] = DEFAULT_PALETTE) -> str:
    """Get a hex color for token+multiplicity using a stable mapping.

    For multiplicity == 0 (standard vocab), returns a neutral gray.
    """
    if multiplicity <= 0:
        return "#333333"  # neutral gray for standard tokens
    idx = _stable_index_for_token(token, multiplicity, len(palette))
    return palette[idx]


def _wrap_markdown(token: str, multiplicity: int) -> str:
    """Apply markdown styling rules by multiplicity.

    Rules:
      0: plain
      1: bold -> **t**
      2: italic -> *t*
      3: strikethrough -> ~~t~~
      4: bold+italic -> ***t***
      5: bold+strikethrough -> ~~**t**~~
      6: italic+strikethrough -> ~~*t*~~
      7+: all three -> ~~***t***~~
    """
    if multiplicity <= 0:
        return token
    if multiplicity == 1:
        return f"**{token}**"
    if multiplicity == 2:
        return f"*{token}*"
    if multiplicity == 3:
        return f"~~{token}~~"
    if multiplicity == 4:
        return f"***{token}***"
    if multiplicity == 5:
        return f"~~**{token}**~~"
    if multiplicity == 6:
        return f"~~*{token}*~~"
    # 7 and above -> all three
    return f"~~***{token}***~~"


def render_markdown(tokens: Sequence[str], multiplicities: Sequence[int]) -> str:
    """Render tokens into markdown string with multiplicity-based styling."""
    assert len(tokens) == len(multiplicities), "tokens and multiplicities must be same length"
    return " ".join(_wrap_markdown(t, int(m)) for t, m in zip(tokens, multiplicities, strict=True))


_LATEX_SPECIALS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def _latex_escape(text: str) -> str:
    out = []
    for ch in text:
        out.append(_LATEX_SPECIALS.get(ch, ch))
    return "".join(out)


def render_latex(
    tokens: Sequence[str], multiplicities: Sequence[int], palette: Sequence[str] = DEFAULT_PALETTE
) -> str:
    """Render tokens into a standalone LaTeX document with color highlighting.

    Uses xcolor and wraps reasoning tokens (mult>0) in \textcolor{<hex>}{token}.
    """
    assert len(tokens) == len(multiplicities), "tokens and multiplicities must be same length"

    body_parts: list[str] = []
    for t, m in zip(tokens, multiplicities, strict=True):
        esc = _latex_escape(t)
        if int(m) <= 0:
            body_parts.append(esc)
        else:
            color = color_for_token(t, int(m), palette)
            # xcolor supports HTML colors via [HTML]{RRGGBB}
            body_parts.append(f"\\textcolor[HTML]{{{color.lstrip('#').upper()}}}{{{esc}}}")
    body = " ".join(body_parts)

    doc = (
        r"\documentclass[preview]{standalone}"
        "\n"
        r"\usepackage[T1]{fontenc}"
        "\n"
        r"\usepackage{lmodern}"
        "\n"
        r"\usepackage{xcolor}"
        "\n"
        r"\begin{document}"
        "\n"
        f"{body}\n"
        r"\end{document}"
        "\n"
    )
    return doc


def _maybe_render_image(
    tokens: Sequence[str],
    multiplicities: Sequence[int],
    out_path: Path,
    palette: Sequence[str] = DEFAULT_PALETTE,
) -> Path | None:
    """Render a simple horizontal text image with colored backgrounds (if MPL present).

    Returns the output path if successful, else None.
    """
    if not HAS_MPL:  # pragma: no cover - exercised in environments w/o MPL
        return None

    # Basic monospace layout approximation
    fig, ax = plt.subplots(figsize=(12, 1.2))
    ax.axis("off")

    x = 0.02
    y = 0.5
    char_w = 0.012  # rough per-character width fraction
    space = 1  # one space between tokens

    for tok, mult in zip(tokens, multiplicities, strict=True):
        tok_str = str(tok)
        width = (len(tok_str) + space) * char_w
        if int(mult) > 0:
            color = color_for_token(tok_str, int(mult), palette)
            bbox = dict(facecolor=color, edgecolor="none", boxstyle="round,pad=0.2")
            ax.text(
                x,
                y,
                tok_str,
                fontsize=12,
                family="monospace",
                va="center",
                bbox=bbox,
                color="black",
            )
        else:
            ax.text(x, y, tok_str, fontsize=12, family="monospace", va="center", color="#222222")
        x += width

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def visualize_reasoning(
    tokens: Sequence[str],
    multiplicities: Sequence[int],
    out_prefix: Path,
    palette: Sequence[str] = DEFAULT_PALETTE,
) -> RenderOutputs:
    """Render all three representations given tokens and multiplicities.

    Args:
        tokens: token strings in order
        multiplicities: same length; 0 for standard, >0 for reasoning variants
        out_prefix: path prefix (e.g., fig/sample) used for outputs
        palette: color palette

    Returns:
        RenderOutputs with paths/content for all formats.
    """
    assert len(tokens) == len(multiplicities), "tokens and multiplicities must be same length"

    img_path = _maybe_render_image(tokens, multiplicities, Path(str(out_prefix) + ".png"), palette)
    md = render_markdown(tokens, multiplicities)
    tex = render_latex(tokens, multiplicities, palette)

    # Persist textual outputs next to image
    md_path = Path(str(out_prefix) + ".md")
    tex_path = Path(str(out_prefix) + ".tex")
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(md)
    tex_path.write_text(tex)

    return RenderOutputs(image_path=img_path, markdown_text=md, latex_text=tex)


def _load_tokens_from_json(json_path: Path) -> tuple[list[str], list[int]]:
    data = json.loads(Path(json_path).read_text())
    tokens = list(map(str, data["tokens"]))
    multiplicities = [int(x) for x in data["multiplicities"]]
    assert len(tokens) == len(multiplicities), "tokens and multiplicities must be same length"
    return tokens, multiplicities


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Visualize reasoning outputs with multiplicities")
    parser.add_argument(
        "--tokens", nargs="*", default=None, help="Tokens to render (space-separated)"
    )
    parser.add_argument(
        "--multiplicities",
        nargs="*",
        type=int,
        default=None,
        help="Multiplicities aligned with --tokens",
    )
    parser.add_argument(
        "--input-json", type=str, default=None, help="JSON file with tokens+multiplicities"
    )
    parser.add_argument(
        "--out-prefix", type=str, default="fig/reasoning_output", help="Output path prefix"
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.input_json is None and (not args.tokens or args.multiplicities is None):
        parser.error("Provide either --input-json or both --tokens and --multiplicities")

    if args.input_json is not None:
        tokens, multiplicities = _load_tokens_from_json(Path(args.input_json))
    else:
        tokens = [str(t) for t in args.tokens]
        multiplicities = [int(m) for m in args.multiplicities]

    out = visualize_reasoning(tokens, multiplicities, Path(args.out_prefix))

    # Print where files were written for convenience
    if out.image_path is not None:
        print(f"Saved image: {out.image_path}")
    print(f"Saved markdown: {Path(args.out_prefix)}.md")
    print(f"Saved LaTeX: {Path(args.out_prefix)}.tex")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI
    raise SystemExit(main())
