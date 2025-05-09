from __future__ import annotations

import logging
import subprocess
import sys

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
    assert "Usage: bruker2csv" in result.stderr


def test_bruker2csv_bad_input_path(tmp_path):
    """Test bruker2csv with a non-existent input path."""
    bad_input_path = tmp_path / "non_existent_path"
    output_csv = tmp_path / "output.csv"
    cmd = [sys.executable, "-m", "spinplots.cli", str(bad_input_path), str(output_csv)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "An error occurred" in result.stderr
