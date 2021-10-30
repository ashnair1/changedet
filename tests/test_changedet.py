#!/usr/bin/env python
"""Tests for `changedet` package."""

import fire

from changedet.pipeline import ChangeDetPipeline


def test_changedet_cli(capsys):
    fire.Fire(ChangeDetPipeline, ["list"])
    captured = capsys.readouterr()
    result = captured.out
    assert "['cva', 'imgdiff', 'ipca', 'irmad']\n" == result
