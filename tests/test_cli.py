from __future__ import annotations

import logging
import subprocess
import sys

import numpy as np
import pandas as pd

from spinplots import calculate_projections
from spinplots.cli import main

DATA_DIR_1D = "data/1D/glycine/pdata/1"


def test_bruker2csv_success(tmp_path):
    """Test successful execution of bruker2csv."""
    output_csv = tmp_path / "output.csv"
    cmd = [sys.executable, "-m", "spinplots.cli", DATA_DIR_1D, str(output_csv)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    logging.info("STDOUT: %s", result.stdout)
    logging.info("STDERR: %s", result.stderr)

    assert result.returncode == 0
    assert output_csv.is_file()
    assert output_csv.stat().st_size > 0


def test_bruker2csv_wrong_args():
    """Test bruker2csv with incorrect number of arguments."""
    cmd = [sys.executable, "-m", "spinplots.cli", DATA_DIR_1D]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "usage: bruker2csv" in result.stderr


def test_bruker2csv_bad_input_path(tmp_path):
    """Test bruker2csv with a non-existent input path."""
    bad_input_path = tmp_path / "non_existent_path"
    output_csv = tmp_path / "output.csv"
    cmd = [sys.executable, "-m", "spinplots.cli", str(bad_input_path), str(output_csv)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "An error occurred" in result.stderr


def test_bruker2csv_prints_confirmation(tmp_path):
    """A successful run says where the file was written."""
    output_csv = tmp_path / "output.csv"
    cmd = [sys.executable, "-m", "spinplots.cli", DATA_DIR_1D, str(output_csv)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert f"Data written to {output_csv}" in result.stdout


def test_bruker2csv_help():
    """--help prints the usage and exits successfully."""
    cmd = [sys.executable, "-m", "spinplots.cli", "--help"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "usage: bruker2csv" in result.stdout
    assert "path_to_data" in result.stdout

def test_2d_csv_no_index_column_and_be_read(
    tmp_path, monkeypatch, capsys
):
    """2D CSVs must start with the F1 and F2 axes."""
    
    f1, f2 = np.meshgrid(np.linspace(0, 30, 4), np.linspace(0, 10, 6), indexing="ij")
    df = pd.DataFrame(
        {
            "13C ppm": f1.flatten(),
            "1H ppm": f2.flatten(),
            "intensity": np.arange(24, dtype=float),
        }
    )
    df.attrs["nmr_dim"] = 2
    monkeypatch.setattr("spinplots.cli.nmr_df", lambda _path: df)

    output_csv = tmp_path / "two_d.csv"
    assert main(["fake/path", str(output_csv)]) == 0
    assert "Data written to" in capsys.readouterr().out

    assert list(pd.read_csv(output_csv).columns) == ["13C ppm", "1H ppm", "intensity"]
    proj_f1, proj_f2 = calculate_projections(str(output_csv))
    assert len(proj_f1) == 4
    assert len(proj_f2) == 6