#!/usr/bin/env python
"""Tests for `changedet` package."""

import os

import fire
import pytest

from changedet.pipeline import ChangeDetPipeline

TEST_DATA = "tests/data"


def test_changedet_cli(capsys):
    fire.Fire(ChangeDetPipeline, ["list"])
    captured = capsys.readouterr()
    result = captured.out
    assert "['cva', 'imgdiff', 'ipca', 'irmad']\n" == result


def test_changedet_missing_inputs(caplog):
    img1 = os.path.join(TEST_DATA, "t1.tif")
    img2 = os.path.join(TEST_DATA, "t2.tiff")
    with pytest.raises(AssertionError):
        fire.Fire(ChangeDetPipeline, ["--algo", "imgdiff", "run", img1, img2])
    assert caplog.records[0].message == "Images not found"
