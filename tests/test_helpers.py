from __future__ import annotations

import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest

from spinplots._helpers import _PACKAGE_DIR, _in_package, _warn
from spinplots.io import read_nmr
from spinplots.plot import bruker1d, bruker2d

DATA_DIR_1D = "data/1D/glycine/pdata/1"
DATA_DIR_2D = "data/2D/16/pdata/1"


@pytest.fixture(autouse=True)
def non_interactive_matplotlib():
    mpl.use("Agg")
    yield
    plt.close("all")


def test_warn_points_at_the_caller():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn("hello", DeprecationWarning)

    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert caught[0].filename == __file__


def test_warning_from_read_nmr_points_at_user_code():
    """The default-tag warning is raised deep inside SpinCollection.append."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        read_nmr(DATA_DIR_1D)

    messages = [w for w in caught if "No tag provided" in str(w.message)]
    assert len(messages) == 1
    assert messages[0].filename == __file__


def test_unknown_kwarg_warning_points_at_user_code_via_plot_method():
    """spins.plot() -> SpinCollection.plot -> bruker1d -> _validate_kwargs."""
    spins = read_nmr(DATA_DIR_1D, tags="glycine")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spins.plot(not_an_option=1, return_fig=True)

    messages = [w for w in caught if "not_an_option" in str(w.message)]
    assert len(messages) == 1
    assert messages[0].filename == __file__


def test_unknown_kwarg_warning_points_at_user_code_when_called_directly():
    spins = read_nmr(DATA_DIR_1D, tags="glycine")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bruker1d([spins.spectrum], not_an_option=1, return_fig=True)

    messages = [w for w in caught if "not_an_option" in str(w.message)]
    assert len(messages) == 1
    assert messages[0].filename == __file__


def test_cmap_warning_message_is_readable():
    spins = read_nmr(DATA_DIR_2D, homonuclear=False, tags="hetcor")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bruker2d(
            [spins.spectrum, spins.spectrum],
            contour_start=1e5,
            cmap=["viridis", "plasma"],
            return_fig=True,
        )

    messages = [str(w.message) for w in caught if "cmap" in str(w.message)]
    assert len(messages) >= 1
    assert "Consider using `color` instead of `cmap` when overlapping" in messages[0]


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        (_PACKAGE_DIR / "plot.py", True),
        (_PACKAGE_DIR / "readers" / "jeol.py", True),  # subpackage
        (_PACKAGE_DIR.with_name(_PACKAGE_DIR.name + "_extra") / "plot.py", False),
        (_PACKAGE_DIR.parent / "test_user.py", False),
        ("<string>", False),
        ("<stdin>", False),
        ("<ipython-input-3-abc123>", False),
    ],
)
def test_in_package(filename, expected):
    assert _in_package(str(filename)) is expected


def test_warn_from_code_without_a_real_filename():
    """exec'd code (REPL, some notebook front ends) has filename '<string>'."""
    source = "from spinplots._helpers import _warn\n_warn('from exec')"
    code = compile(source, "<string>", "exec")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        exec(code, {})

    assert len(caught) == 1
    assert caught[0].filename == "<string>"
