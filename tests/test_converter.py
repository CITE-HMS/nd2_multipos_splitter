# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pytest",
#     "nd2==0.11.3",
#     "dask==2026.3.0",
#     "tifffile==2026.5.15",
#     "ome-types==0.6.3",
#     "pyqt6==6.11.0",
#     "superqt==0.8.2",
#     "tqdm==4.67.3",
#     "rich==15.0.0",
#     "numpy",
# ]
# ///
"""Tests for the ND2 -> OME-TIFF converter.

Run with: ``uv run tests/test_converter.py``

Test data lives in ``tests/data`` (small synthetic ND2 samples bundled with
the repo).
"""

from __future__ import annotations

import importlib.util
import threading
from pathlib import Path

import nd2
import numpy as np
import pytest
import tifffile
from ome_types import from_xml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CONVERTER = ROOT / "nd2_multipos_splitter.py"

# import the converter script by path (it is a uv script, not a package)
_spec = importlib.util.spec_from_file_location("nd2_multipos_splitter", CONVERTER)
conv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(conv)

DATA = HERE / "data"


def _sample(name: str) -> Path:
    path = DATA / name
    if not path.exists():
        pytest.skip(f"missing test data {name}")
    return path


def _layout(src: Path) -> tuple[int, list[str], int | None]:
    """Return (num_positions, position_names, p_axis) for ``src``."""
    with nd2.ND2File(src) as f:
        num_p = f.sizes.get("P", 1)
        positions = conv._get_positions(f)
        names = [conv._get_position_name(positions, i) for i in range(num_p)]
        p_ax = list(f.sizes).index("P") if "P" in f.sizes else None
    return num_p, names, p_ax


def _plane_count(src: Path) -> int:
    with nd2.ND2File(src) as f:
        s = f.sizes
        return s.get("T", 1) * s.get("Z", 1) * s.get("C", 1)


def _pos_slice(full: np.ndarray, p_ax: int | None, index: int) -> np.ndarray:
    if p_ax is None:
        return full
    return full[(slice(None),) * p_ax + (index,)]


# files exercised by the main correctness test
CORRECTNESS_FILES = [
    "dims_p4z5t3c2y32x32.nd2",
    "dims_p2z5t3-2c4y32x32.nd2",
    "dims_p1z5t3c2y32x32.nd2",
    "dims_z5t3c2y32x32.nd2",
    "dims_t3c2y32x32.nd2",
    "dims_c2y32x32.nd2",
]


@pytest.fixture(params=CORRECTNESS_FILES, ids=lambda p: p)
def src(request) -> Path:
    return _sample(request.param)


def test_file_count_and_names(src: Path, tmp_path: Path) -> None:
    num_p, names, _ = _layout(src)
    list(conv.convert_nd2_to_ome_tiff(src, tmp_path, workers=2))

    written = sorted(tmp_path.rglob("*.ome.tif"))
    assert len(written) == num_p

    stem = src.stem
    if num_p == 1:
        assert {p.name for p in written} == {f"{stem}.ome.tif"}
    else:
        assert {p.name for p in written} == {
            f"{stem}_{n}.ome.tif" for n in names
        }


def test_metadata_and_pixels_roundtrip(src: Path, tmp_path: Path) -> None:
    num_p, names, p_ax = _layout(src)
    expected_planes = _plane_count(src)
    full = nd2.imread(str(src))

    list(conv.convert_nd2_to_ome_tiff(src, tmp_path, workers=3))

    for index in range(num_p):
        out = conv._get_filepath(
            src.stem, tmp_path, names[index], well_folders=False, num_p=num_p
        )
        with tifffile.TiffFile(out) as tif:
            ome = from_xml(tif.ome_metadata)
            data = tif.asarray()

        assert len(ome.images) == 1
        pixels = ome.images[0].pixels
        assert len(pixels.planes) == expected_planes
        assert pixels.type.value == str(full.dtype)

        ref = _pos_slice(full, p_ax, index)
        assert np.array_equal(data.reshape(ref.shape), ref)


def test_workers_produce_identical_output(tmp_path: Path) -> None:
    src = _sample("dims_p4z5t3c2y32x32.nd2")
    _, names, _ = _layout(src)

    out1 = tmp_path / "w1"
    out4 = tmp_path / "w4"
    out1.mkdir()
    out4.mkdir()
    list(conv.convert_nd2_to_ome_tiff(src, out1, workers=1))
    list(conv.convert_nd2_to_ome_tiff(src, out4, workers=4))

    for name in names:
        f = f"{src.stem}_{name}.ome.tif"
        assert np.array_equal(tifffile.imread(out1 / f), tifffile.imread(out4 / f))


def test_well_folders_groups_files(tmp_path: Path) -> None:
    # position names look like "name_A01_0000" -> well folder "name_A01"
    src = _sample("JOBS_Platename_WellA01_ChannelWidefield_Green_Seq0000.nd2")
    num_p, names, _ = _layout(src)

    list(conv.convert_nd2_to_ome_tiff(src, tmp_path, well_folders=True, workers=2))

    for name in names:
        well = name.rsplit("_", 1)[0]
        assert (tmp_path / well / f"{src.stem}_{name}.ome.tif").exists()
    assert len(list(tmp_path.rglob("*.ome.tif"))) == num_p


def test_well_folders_bad_names_raise_before_writing(tmp_path: Path) -> None:
    # position names like "A01" have no "_<digits>" suffix -> must raise, write nothing
    src = _sample("wellplate96_4_wells_without_jobs.nd2")
    with pytest.raises(ValueError, match="well folders"):
        list(conv.convert_nd2_to_ome_tiff(src, tmp_path, well_folders=True))
    assert list(tmp_path.rglob("*.ome.tif")) == []


def test_unsupported_axes_raise(tmp_path: Path) -> None:
    src = _sample("dims_rgb.nd2")  # has an RGB 'S' axis
    with pytest.raises(ValueError, match="Cannot write OME-TIFF"):
        list(conv.convert_nd2_to_ome_tiff(src, tmp_path))
    assert list(tmp_path.rglob("*.ome.tif")) == []


def test_single_position_drops_suffix(tmp_path: Path) -> None:
    src = _sample("dims_c2y32x32.nd2")  # no P axis
    list(conv.convert_nd2_to_ome_tiff(src, tmp_path))
    written = list(tmp_path.rglob("*.ome.tif"))
    assert [p.name for p in written] == [f"{src.stem}.ome.tif"]


def test_preset_stop_event_writes_nothing(tmp_path: Path) -> None:
    src = _sample("dims_p4z5t3c2y32x32.nd2")
    event = threading.Event()
    event.set()
    list(conv.convert_nd2_to_ome_tiff(src, tmp_path, stop_event=event))
    assert list(tmp_path.rglob("*.ome.tif")) == []


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
