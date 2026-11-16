"""How variant names are read (pipeline/run.py). No data needed.

A variant name is "<base>-<option>-<option>...". Most options switch a step off; "min25" adds one.
These tests pin that behaviour down so a later change cannot silently ignore an option.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline.run import switched_off, variant_options  # noqa: E402


def test_no_options():
    assert variant_options("hybrid_tree_tidy") == []


def test_one_option_still_works_as_before():
    assert variant_options("hybrid_tree-ball_missing_slow") == ["ball_missing_slow"]
    assert variant_options("hybrid_tree_restart-literal_corner") == ["literal_corner"]


def test_several_options_can_be_chained():
    assert variant_options("hybrid_stoppage_soft-midpoint-edges") == ["midpoint", "edges"]


def test_min25_is_an_add_on_not_a_switch_off():
    opts = variant_options("hybrid_tree_tidy-merge-min25")
    assert "min25" in opts
    assert switched_off(opts) == ("merge",)
