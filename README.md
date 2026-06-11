# nd2 Multiposition Splitter

Download the **`nd2_multipos_splitter.py`** file: right-click [here](https://raw.githubusercontent.com/CITE-HMS/nd2_multipos_splitter/main/nd2_multipos_splitter.py) and choose "Save Link As...".

A small GUI tool that splits a multiposition `.nd2` file into one
**OME-TIFF** per stage position. It works with plain XY position lists,
plate/well experiments, and single-position files.

## What it does

- Reads a multiposition `.nd2` file and writes one `.ome.tif` per position
  (a single-position file is written without a position suffix).
- Optionally groups each position's file into a `<well>/` subfolder
  (**create well folders**), based on position names ending in
  `_<digits>` (e.g. `A1_000`, `name_A01_0000`).
- Writes positions in parallel (**Parallel writers**) for faster conversion
  on multi-core machines / fast disks.
- Shows progress and lets you stop the conversion early; files already being
  written are completed before stopping.

Output naming:

- Multiposition: `<file>_<position>.ome.tif` (inside `<well>/` if well
  folders are enabled).
- Single position: `<file>.ome.tif`.

## How to run

### Option 1: with `pyrunner` (no terminal needed)

[`pyrunner`](https://github.com/fdrgsp/pyrunner) lets you run a `uv` Python
script by simply double-clicking it.

1. Download and install `pyrunner` from
   [github.com/fdrgsp/pyrunner](https://github.com/fdrgsp/pyrunner).
2. Download [`nd2_multipos_splitter.py`](https://raw.githubusercontent.com/CITE-HMS/nd2_multipos_splitter/main/nd2_multipos_splitter.py).
3. Double-click on the `pyrunner` icon and select `nd2_multipos_splitter.py` to run it.

### Option 2: with `uv`

[`uv`](https://docs.astral.sh/uv/) reads the dependencies declared at the top
of the script and runs it in an isolated environment — no manual setup
needed.

1. [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/) if
   you don't have it yet.
2. Download [`nd2_multipos_splitter.py`](https://raw.githubusercontent.com/CITE-HMS/nd2_multipos_splitter/main/nd2_multipos_splitter.py).
3. From a terminal, in the folder containing the script, run:

   ```sh
   uv run nd2_multipos_splitter.py
   ```
