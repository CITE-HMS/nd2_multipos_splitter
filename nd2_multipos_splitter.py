# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "nd2==0.11.3",
#     "dask==2026.3.0",
#     "tifffile==2026.5.15",
#     "ome-types==0.6.3",
#     "pyqt6==6.11.0",
#     "superqt==0.8.2",
#     "tqdm==4.67.3",
#     "rich==15.0.0",
# ]
# ///
"""Split a multiposition ND2 file into one OME-TIFF per position.

Run with: ``uv run nd2_multipos_splitter.py``

The conversion core (`convert_nd2_to_ome_tiff`) is GUI-independent so it can be
called from scripts/tests; `FileDialog` is a thin PySide6 wrapper around it.
"""

from __future__ import annotations

import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import dask.array as da
import nd2
import tifffile as tf
from nd2.structures import Position, XYPosLoop
from ome_types.model import OME, TiffData
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from rich import print
from superqt.utils import create_worker
from tqdm import tqdm

STYLESHEET = """
    QProgressBar {
        text-align: center;
        border: 2px;
        border-radius: 5px;
}
"""
FIXED_SIZE_POLICY = (QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
SUPPORTED_AXES = ("X", "Y", "Z", "C", "P", "T")
# position name accepted for well folders, e.g. "A1_000", "name_A01_0000"
WELL_NAME_RE = re.compile(r".+_\d+")
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)

INFO_TEXT = """\
<h3>ND2 to OME-TIFF</h3>
<p>Splits a multiposition <b>.nd2</b> file into one <b>OME-TIFF</b> per stage
position (works for plain XY position lists, plate/well experiments, and
single-position files).</p>

<p><b>How to run</b><br>
From a terminal in this folder: <code>uv run nd2_multipos_splitter.py</code></p>

<p><b>Options</b></p>
<ul>
<li><b>Input File</b> — the <code>.nd2</code> file to split.</li>
<li><b>Output Folder</b> — where the OME-TIFFs are written.</li>
<li><b>create well folders</b> — group each position's file inside a subfolder
named after its well. Requires position names ending in
<code>_&lt;digits&gt;</code> (e.g. <code>A1_000</code>, <code>name_A01_0000</code>);
the well folder is everything before the last <code>_</code>. If any position
name does not match, conversion stops with an error before writing.</li>
<li><b>Parallel writers</b> — number of positions written at the same time.
Higher is faster on multi-core machines / fast disks; lower uses less memory.</li>
<li><b>Stop</b> — stops queuing new positions; files already being written finish.</li>
</ul>

<p><b>Output names</b><br>
Multiposition: <code>&lt;file&gt;_&lt;position&gt;.ome.tif</code>
(inside <code>&lt;well&gt;/</code> when well folders are enabled).<br>
Single position: <code>&lt;file&gt;.ome.tif</code>.</p>
"""


def _get_positions(f: nd2.ND2File) -> list[Position]:
    """Return the XYPosLoop points, or an empty list if there is no position loop."""
    return next(
        (
            exp.parameters.points
            for exp in f.experiment
            if isinstance(exp, XYPosLoop)
        ),
        [],
    )


def _get_position_name(positions: list[Position], index: int) -> str:
    """Return the position name, falling back to ``pos_<index>``."""
    if index < len(positions) and positions[index].name:
        return positions[index].name
    return f"pos_{index:03d}"


def _get_filepath(
    stem: str, dest: Path, pos_name: str, *, well_folders: bool, num_p: int
) -> Path:
    """Build the output path. Single-position files drop the position suffix."""
    if num_p == 1:
        return dest / f"{stem}.ome.tif"
    if well_folders:
        # pos_name validated as "<well>_<digits>" before conversion starts
        well = pos_name.rsplit("_", 1)[0]
        folder = dest / well
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{stem}_{pos_name}.ome.tif"
    return dest / f"{stem}_{pos_name}.ome.tif"


def _position_ome(ome: OME, index: int) -> OME:
    """Return a single-image OME for position ``index``.

    nd2 (>=0.11) already emits one ``Image`` per position with the correct
    planes, dtype and dimension order; we only need to flip the pixels from
    ``metadata_only`` to a tiff-data block so tifffile writes valid OME-TIFF.
    """
    image = ome.images[index]
    pixels = image.pixels.model_copy(
        update={
            "id": "Pixels:0",
            "metadata_only": None,
            "tiff_data_blocks": [TiffData(plane_count=len(image.pixels.planes))],
        }
    )
    image = image.model_copy(update={"id": "Image:0", "pixels": pixels})
    return ome.model_copy(update={"images": [image]})


def convert_nd2_to_ome_tiff(
    input_file: Path,
    dest: Path,
    *,
    well_folders: bool = False,
    workers: int = DEFAULT_WORKERS,
    stop_event: threading.Event | None = None,
    test: bool = False,
) -> Iterator[tuple[int, int]]:
    """Split ``input_file`` into one OME-TIFF per position inside ``dest``.

    Yields ``(done, remaining)`` after each position is written so a GUI can
    drive a progress bar. Writing is parallelised across ``workers`` threads,
    each with its own ND2 reader handle (the SDK reader is not thread-safe).

    Parameters
    ----------
    input_file, dest : Path
        Source ``.nd2`` file and existing destination directory.
    well_folders : bool
        Group each position inside a ``<well>/`` subfolder. Requires every
        position name to match ``<well>_<digits>``.
    workers : int
        Number of positions written concurrently.
    stop_event : threading.Event | None
        If set (before or during the run), no further positions are queued.
    test : bool
        Print target paths instead of writing files.
    """
    stop_event = stop_event or threading.Event()

    with nd2.ND2File(input_file) as f:
        unsupported = [s for s in f.sizes if s not in SUPPORTED_AXES]
        if unsupported:
            raise ValueError(
                f"Cannot write OME-TIFF with size(s) {unsupported}. Only "
                f"{', '.join(SUPPORTED_AXES)} are currently supported."
            )

        num_p = f.sizes.get("P", 1)
        positions = _get_positions(f)
        names = [_get_position_name(positions, i) for i in range(num_p)]
        p_ax = list(f.sizes).index("P") if "P" in f.sizes else None
        stem = input_file.stem

        # validate well-folder names up front so we fail before writing anything
        if well_folders and num_p > 1:
            bad = [n for n in names if not WELL_NAME_RE.fullmatch(n)]
            if bad:
                raise ValueError(
                    "To create well folders, every position name must end with "
                    "'_<number>' (e.g. 'A1_000', 'name_A01_0000'). These do not: "
                    f"{bad}. Uncheck 'create well folders' to convert this file."
                )

        # nd2's ome_metadata + dimension_order already match to_dask()'s axis order
        ome = f.ome_metadata()

    # one reader (and dask array) per worker thread; the SDK reader is not
    # thread-safe, so sharing a single handle would corrupt reads
    tls = threading.local()
    readers: list[nd2.ND2File] = []
    readers_lock = threading.Lock()

    def _thread_dask() -> da.Array:
        dask = getattr(tls, "dask", None)
        if dask is None:
            reader = nd2.ND2File(input_file)
            with readers_lock:
                readers.append(reader)
            dask = tls.dask = reader.to_dask()
        return dask

    def _write_position(index: int) -> Path | None:
        if stop_event.is_set():
            return None
        dask = _thread_dask()
        data = dask if p_ax is None else dask[(slice(None),) * p_ax + (index,)]
        filepath = _get_filepath(
            stem, dest, names[index], well_folders=well_folders, num_p=num_p
        )
        if test:
            print(filepath)
            time.sleep(0.2)
            return filepath
        xml = _position_ome(ome, index).to_xml().encode("utf-8")
        with tf.TiffWriter(filepath, bigtiff=True) as writer:
            writer.write(
                data.compute(),
                description=xml,
                metadata=None,
                photometric="minisblack",
            )
        return filepath

    workers = max(1, min(workers, num_p))
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_write_position, i): i for i in range(num_p)}
            done = 0
            for future in tqdm(as_completed(futures), total=num_p):
                future.result()  # re-raise any worker error
                done += 1
                yield done, num_p - done
                if stop_event.is_set():
                    executor.shutdown(cancel_futures=True)
                    break
    finally:
        for reader in readers:
            reader.close()


class FileDialog(QWidget):
    def __init__(self, parent: QWidget | None = None, *, test: bool = False):
        super().__init__(parent)

        self._stop_event = threading.Event()
        self._test = test  # run without saving the files, just printing the path

        self.setWindowTitle("ND2 to OME-TIFF")

        # file
        file_wdg = QWidget()
        file_layout = QHBoxLayout(file_wdg)
        file_layout.setContentsMargins(0, 0, 0, 0)
        file_layout.setSpacing(5)
        file_lbl = QLabel("Input File:")
        file_lbl.setSizePolicy(*FIXED_SIZE_POLICY)
        self.file_le = QLineEdit()
        self.file_le.setReadOnly(True)
        self._file_btn = QPushButton("Browse")
        self._file_btn.clicked.connect(self._select_input_file)
        file_layout.addWidget(file_lbl)
        file_layout.addWidget(self.file_le)
        file_layout.addWidget(self._file_btn)

        # dest
        dest_wdg = QWidget()
        dest_layout = QHBoxLayout(dest_wdg)
        dest_layout.setContentsMargins(0, 0, 0, 0)
        dest_layout.setSpacing(5)
        dest_lbl = QLabel("Output Folder:")
        dest_lbl.setSizePolicy(*FIXED_SIZE_POLICY)
        self.dest_le = QLineEdit()
        self.dest_le.setReadOnly(True)
        self._dest_btn = QPushButton("Browse")
        self._dest_btn.clicked.connect(self._select_output_folder)
        dest_layout.addWidget(dest_lbl)
        dest_layout.addWidget(self.dest_le)
        dest_layout.addWidget(self._dest_btn)

        # options: well folders + parallel writers
        opts_wdg = QWidget()
        opts_layout = QHBoxLayout(opts_wdg)
        opts_layout.setContentsMargins(0, 0, 0, 0)
        opts_layout.setSpacing(5)
        self.well_folder_cbox = QCheckBox(text="create well folders")
        self.well_folder_cbox.setToolTip(
            "If checked, each file is saved in a folder named after its well. "
            "Position names must end with '_<number>', like 'A1_000', 'A1_001'."
        )
        workers_lbl = QLabel("Parallel writers:")
        workers_lbl.setSizePolicy(*FIXED_SIZE_POLICY)
        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(1, os.cpu_count() or 1)
        self.workers_spin.setValue(DEFAULT_WORKERS)
        self.workers_spin.setToolTip("Number of positions written at the same time.")
        opts_layout.addWidget(self.well_folder_cbox)
        opts_layout.addStretch()
        opts_layout.addWidget(workers_lbl)
        opts_layout.addWidget(self.workers_spin)

        # run / stop / info buttons
        btns_wdg = QWidget()
        btns_layout = QHBoxLayout(btns_wdg)
        btns_layout.setContentsMargins(0, 5, 0, 0)
        btns_layout.setSpacing(5)
        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.info_btn = QPushButton("?")
        self.info_btn.setSizePolicy(*FIXED_SIZE_POLICY)
        self.info_btn.setToolTip("How to use this tool")
        self.run_btn.clicked.connect(self._on_run)
        self.stop_btn.clicked.connect(self._on_stop)
        self.info_btn.clicked.connect(self._show_info)
        btns_layout.addWidget(self.run_btn)
        btns_layout.addWidget(self.stop_btn)
        btns_layout.addWidget(self.info_btn)

        # progress bar
        prg_bar_wdg = QWidget()
        prg_bar_layout = QHBoxLayout(prg_bar_wdg)
        prg_bar_layout.setContentsMargins(5, 5, 5, 5)
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet(STYLESHEET)
        self.progress_bar.setValue(0)
        self.status_lbl = QLabel("Stopped.")
        prg_bar_layout.addWidget(self.progress_bar)
        prg_bar_layout.addWidget(self.status_lbl)

        # main
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(5)
        main_layout.addWidget(file_wdg)
        main_layout.addWidget(dest_wdg)
        main_layout.addWidget(opts_wdg)
        main_layout.addWidget(btns_wdg)
        main_layout.addWidget(prg_bar_wdg)

        # set label sizes
        file_lbl.setFixedWidth(dest_lbl.sizeHint().width())

    def _enable(self, state: bool) -> None:
        self.file_le.setEnabled(state)
        self.dest_le.setEnabled(state)
        self._file_btn.setEnabled(state)
        self._dest_btn.setEnabled(state)
        self.well_folder_cbox.setEnabled(state)
        self.workers_spin.setEnabled(state)
        self.run_btn.setEnabled(state)

    def _select_input_file(self) -> None:
        """Select nd2 file."""
        file, _ = QFileDialog.getOpenFileName(
            self, "Select nd2 File", "", "nd2 (*.nd2)"
        )
        if file:
            self.file_le.setText(file)

    def _select_output_folder(self) -> None:
        """Select output folder."""
        if folder := QFileDialog.getExistingDirectory(self, "Select Output Folder"):
            self.dest_le.setText(folder)

    def _show_info(self) -> None:
        """Show usage instructions."""
        box = QMessageBox(self)
        box.setWindowTitle("ND2 to OME-TIFF - Info")
        box.setTextFormat(Qt.TextFormat.RichText)
        box.setText(INFO_TEXT)
        box.exec()

    def _on_run(self) -> None:
        """Validate inputs, prepare the UI, and start the conversion worker."""
        if not self.file_le.text() or not self.dest_le.text():
            return

        input_file = Path(self.file_le.text())
        dest = Path(self.dest_le.text())
        if not input_file.exists() or not dest.exists():
            QMessageBox.critical(
                self, "Conversion failed", "`Input File` or `Output Directory` do not exist."
            )
            return

        with nd2.ND2File(input_file) as f:
            num_p = f.sizes.get("P", 1)

        self._enable(False)
        self.progress_bar.setRange(0, num_p)
        self.status_lbl.setText(f"Running... ({num_p} files left).")

        self._stop_event.clear()
        create_worker(
            self._split_nd2,
            input_file,
            dest,
            self.well_folder_cbox.isChecked(),
            self.workers_spin.value(),
            _start_thread=True,
            _connect={
                "yielded": self._update_progress_bar,
                "errored": self._on_error,
                "finished": self._on_stop,
            },
        )

    def _split_nd2(
        self, input_file: Path, dest: Path, well_folders: bool, workers: int
    ) -> Iterator[tuple[int, int]]:
        """Delegate to `convert_nd2_to_ome_tiff`."""
        yield from convert_nd2_to_ome_tiff(
            input_file,
            dest,
            well_folders=well_folders,
            workers=workers,
            stop_event=self._stop_event,
            test=self._test,
        )

    def _update_progress_bar(self, args: tuple[int, int]) -> None:
        index, remaining_files = args
        self.progress_bar.setValue(index)
        self.status_lbl.setText(f"Running... ({remaining_files} files left).")

    def _on_error(self, exc: Exception) -> None:
        self._on_stop()
        QMessageBox.critical(self, "Conversion failed", str(exc))

    def _on_stop(self) -> None:
        self._stop_event.set()
        self._enable(True)
        self.status_lbl.setText("Stopped.")
        self.progress_bar.setValue(0)


if __name__ == "__main__":
    import sys

    from PyQt6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    wdg = FileDialog(test=False)
    wdg.show()
    sys.exit(app.exec())
