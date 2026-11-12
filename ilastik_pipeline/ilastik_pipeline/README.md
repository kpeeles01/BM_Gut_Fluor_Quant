# Automated ilastik SumIP Pipeline

Automates the validated workflow from **SumIP TIFF files through Steps 1-5**.

## What it does
1. Step 1 ilastik: SumIP -> Simple Segmentation HDF5.
2. Step 2 Python: converts segmentation to binary, 3x3 dilation (3 iterations), subtracts original segmentation, saves BM-map TIFF and HDF5 (`bm_map_img`).
3. Step 3 ilastik: Raw SumIP + matched BM map -> object-measurement CSV.
4. Step 4 pt1 ilastik: Raw SumIP -> prediction probabilities HDF5.
5. Step 4 pt2 ilastik: Raw SumIP + matched probabilities -> background-measurement CSV.
6. Step 5 Python: summarizes BM and IBG CSVs into one Excel workbook.

The four trained `.ilp` files are stored in `models/` and are not retrained by this script.

## Install
Install ilastik (the projects were created with ilastik 1.4.1b20) and Python 3. Then in this repository:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run on Windows
Put all SumIP TIFFs for one experiment in one folder, then:

```bash
python run_pipeline.py "C:\path\to\SumIP"
```

If ilastik is not found automatically:

```bash
python run_pipeline.py "C:\path\to\SumIP" --ilastik "C:\Program Files\ilastik-1.4.1b20\ilastik.exe"
```

To resume after an interrupted run and reuse completed files:

```bash
python run_pipeline.py "C:\path\to\SumIP" --skip-existing
```

## Output
A sibling folder named `<SumIP folder>_pipeline` is created with:

- `01_segmentations/`
- `02_bm_maps/`
- `03_measurements/`
- `04_background_predictions/`
- `05_background_measurements/`
- `Results/analysis_results.xlsx`

## Important validation note
The first automated version intentionally preserves the Step 2 notebook's RGBA BM-map representation because the Step 3 ilastik project was trained with a `y,x,c` BM-map input with 4 channels. Validate automated results against a previously processed experiment before replacing the manual workflow.
