import subprocess
import sys
import pytest

DATA_DIR_1D = "data/1D/8/pdata/1"


def test_bruker2csv_success(tmp_path):
    """Test successful execution of bruker2csv."""
    output_csv = tmp_path / "output.csv"
    cmd = [sys.executable, "-m", "spinplots.cli", DATA_DIR_1D, str(output_csv)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)

    assert result.returncode == 0
    assert output_csv.is_file()
    assert "Data written to" in result.stdout
    assert output_csv.stat().st_size > 0


def test_bruker2csv_wrong_args(tmp_path):
    """Test bruker2csv with incorrect number of arguments."""
    cmd = [sys.executable, "-m", "spinplots.cli", DATA_DIR_1D]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "Usage: bruker2csv" in result.stdout


def test_bruker2csv_bad_input_path(tmp_path):
    """Test bruker2csv with a non-existent input path."""
    output_csv = tmp_path / "output.csv"
    bad_input_path = tmp_path / "non_existent_dir"
    cmd = [sys.executable, "-m", "spinplots.cli", str(bad_input_path), str(output_csv)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "An error occurred" in result.stdout
