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


def test_changedet_irmad(caplog):
    img1 = os.path.join(TEST_DATA, "t1.tif")
    img2 = os.path.join(TEST_DATA, "t2.tif")
    niter = 2
    fire.Fire(
        ChangeDetPipeline, ["--algo", "irmad", "run", img1, img2, "--niter", str(niter)]
    )
    assert (
        caplog.records[0].message
        == f"Running IRMAD algorithm for {niter} iteration(s) with significance level 0.000100"
    )
    assert caplog.records[1].message == "Change map written to irmad_cmap.tif"


def test_changedet_cva(caplog):
    # TODO: Add distance option
    img1 = os.path.join(TEST_DATA, "t1.tif")
    img2 = os.path.join(TEST_DATA, "t2.tif")
    fire.Fire(ChangeDetPipeline, ["--algo", "cva", "run", img1, img2])
    assert caplog.records[0].message == "Calculating change vectors"
    assert caplog.records[1].message == "Change map written to cva_cmap.tif"


def test_changedet_ipca(caplog):
    img1 = os.path.join(TEST_DATA, "t1.tif")
    img2 = os.path.join(TEST_DATA, "t2.tif")
    niter = 2
    band = 1
    fire.Fire(
        ChangeDetPipeline,
        ["--algo", "ipca", "run", img1, img2, "--niter", str(niter), "--band", str(band)],
    )
    assert (
        caplog.records[0].message
        == f"Running IPCA algorithm for {niter} iteration(s) on band {band}"
    )
    assert caplog.records[1].message == f"Processing band {band}"
    assert caplog.records[2].message == "Change map written to ipca_cmap.tif"
