"""
Tests for viz/visualize_reasoning.py formatting and rendering helpers.

Covers:
- Markdown formatting rules by multiplicity
- Stable color hashing behavior
- LaTeX escaping and color wrapping
- Optional image rendering write-out (skipped if matplotlib unavailable)
"""

from __future__ import annotations

from pathlib import Path

import pytest

from viz.visualize_reasoning import (
    color_for_token,
    render_latex,
    render_markdown,
)


class TestMarkdownFormatting:
    def test_markdown_rules(self):
        tokens = ["t0", "t1", "t2", "t3", "t4", "t5", "t6", "t7"]
        multiplicities = list(range(8))
        md = render_markdown(tokens, multiplicities)
        # Check each token formatting fragment exists as expected
        assert "t0" in md  # plain
        assert "**t1**" in md  # bold
        assert "*t2*" in md  # italic
        assert "~~t3~~" in md  # strikethrough
        assert "***t4***" in md  # bold+italic
        assert "~~**t5**~~" in md  # bold+strikethrough
        assert "~~*t6*~~" in md  # italic+strikethrough
        assert "~~***t7***~~" in md  # all three

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            render_markdown(["a"], [])


class TestColorMapping:
    def test_color_stability(self):
        c1 = color_for_token("hello", 3)
        c2 = color_for_token("hello", 3)
        assert c1 == c2

    def test_color_changes_with_multiplicity(self):
        # Not guaranteed different with a tiny palette, but very likely.
        # Use a larger custom palette to ensure difference.
        big_palette = tuple(f"#{i:06x}" for i in range(1, 64))
        c1b = color_for_token("hello", 1, big_palette)
        c2b = color_for_token("hello", 2, big_palette)
        assert c1b != c2b

    def test_standard_vocab_color_is_neutral(self):
        assert color_for_token("std", 0) == "#333333"
        assert color_for_token("std", -1) == "#333333"


class TestLatexRendering:
    def test_latex_escaping_and_wrapping(self):
        tokens = ["a_#%$", "b&{}", "c~^\\"]
        multiplicities = [0, 1, 2]
        tex = render_latex(tokens, multiplicities)
        # Ensure document structure
        assert "\\begin{document}" in tex and "\\end{document}" in tex
        # Escaping expectations for token 0 (no color)
        assert "a\\_\\#\\%\\$" in tex
        # Token 1 should be wrapped in color macro
        assert "\\textcolor[HTML]" in tex

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            render_latex(["a"], [])


class TestImageRendering:
    def test_image_write(self, tmp_path: Path):
        pytest.importorskip("matplotlib", reason="matplotlib not installed")
        from viz.visualize_reasoning import visualize_reasoning

        tokens = ["alpha", "beta", "gamma"]
        multiplicities = [0, 1, 2]
        out_prefix = tmp_path / "sample"
        out = visualize_reasoning(tokens, multiplicities, out_prefix)
        assert out.image_path is not None and out.image_path.exists()
        # Also ensure md/tex were written
        assert (tmp_path / "sample.md").exists()
        assert (tmp_path / "sample.tex").exists()
