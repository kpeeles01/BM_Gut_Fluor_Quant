#!/usr/bin/env python3
from __future__ import annotations
import argparse, subprocess, sys, shutil
from pathlib import Path
import h5py
import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / 'models'


def run(cmd):
    print('\n>', ' '.join(str(x) for x in cmd))
    subprocess.run([str(x) for x in cmd], check=True)


def find_ilastik(value: str | None) -> Path:
    candidates = []
    if value: candidates.append(Path(value))
    exe = shutil.which('ilastik.exe') or shutil.which('ilastik')
    if exe: candidates.append(Path(exe))
    candidates += [Path(r'C:\Program Files\ilastik-1.4.1b20\ilastik.exe'), Path(r'C:\Program Files\ilastik-1.4.1\ilastik.exe')]
    for p in candidates:
        if p.exists(): return p
    raise FileNotFoundError('Could not find ilastik. Pass --ilastik "C:\\...\\ilastik.exe".')


def samples(input_dir: Path):
    files = sorted([*input_dir.glob('*.tif'), *input_dir.glob('*.tiff')])
    if not files: raise FileNotFoundError(f'No .tif/.tiff SumIP files found in {input_dir}')
    stems = [p.stem for p in files]
    if len(stems) != len(set(stems)): raise ValueError('Duplicate TIFF stems detected.')
    return files


def h5_dataset(path: Path):
    with h5py.File(path, 'r') as f:
        keys = list(f.keys())
        if not keys: raise ValueError(f'No dataset in {path}')
        key = 'exported_data' if 'exported_data' in f else keys[0]
        return np.asarray(f[key])


def make_bm_map(seg_h5: Path, out_tiff: Path, out_h5: Path, expected_hw: tuple[int,int]):
    data = h5_dataset(seg_h5)
    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(f'Expected Step 1 segmentation to squeeze to 2D; got {data.shape} in {seg_h5.name}')
    # Preserve the notebook's transformation exactly: class labels 1/2 -> binary 1/0.
    binary = np.abs(data.astype(np.int16) - 2).astype(np.uint8)
    dilated = cv2.dilate(binary, np.ones((3,3), np.uint8), iterations=3)
    bm = dilated - binary

    # Preserve Step 3 training representation: matplotlib-generated RGBA TIFF.
    # Compute figure size so output pixels exactly match the source image dimensions.
    h, w = expected_hw
    if bm.shape != (h, w):
        raise ValueError(f'BM map {bm.shape} does not match SumIP {(h,w)} for {seg_h5.name}')
    dpi = 100
    fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi, frameon=False)
    ax = fig.add_axes([0,0,1,1]); ax.imshow(bm, cmap='gray', vmin=0, vmax=1); ax.axis('off')
    fig.savefig(out_tiff, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig)
    rgba = plt.imread(out_tiff)
    if rgba.shape[:2] != (h,w) or rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError(f'Generated BM TIFF has unexpected shape {rgba.shape}; expected {(h,w,4)}')
    with h5py.File(out_h5, 'w') as f:
        f.create_dataset('bm_map_img', data=rgba, compression='gzip')


def image_hw(tif: Path):
    with Image.open(tif) as im:
        return im.height, im.width


def summarize_csv_folder(folder: Path, pred_class: str, label: str):
    rows = []
    for csv in sorted(folder.glob('*.csv')):
        df = pd.read_csv(csv)
        if 'Predicted Class' not in df.columns: continue
        wanted = df.loc[df['Predicted Class'] == pred_class]
        row = {'file name': csv.name, 'predicted class': pred_class,
               'total size in pixels': wanted['Size in pixels'].sum() if 'Size in pixels' in wanted else np.nan}
        for col in ('Total Intensity_1','Total Intensity_2'):
            if col in wanted.columns:
                total = wanted[col].sum()
                row[f'total {col.lower().replace("_", " ")}'] = total
                size = row['total size in pixels']
                row[f'{col} per pixel'] = total/size if pd.notna(size) and size != 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows), label


def main():
    ap = argparse.ArgumentParser(description='Automate SumIP -> ilastik Steps 1-5.')
    ap.add_argument('input_dir', type=Path, help='Folder containing SumIP .tif/.tiff files')
    ap.add_argument('--output-dir', type=Path, default=None)
    ap.add_argument('--ilastik', default=None, help='Path to ilastik.exe')
    ap.add_argument('--skip-existing', action='store_true', help='Reuse existing outputs when present')
    args = ap.parse_args()
    input_dir = args.input_dir.resolve(); out = (args.output_dir or input_dir.parent / f'{input_dir.name}_pipeline').resolve()
    ilastik = find_ilastik(args.ilastik)
    imgs = samples(input_dir)

    dirs = {k: out/k for k in ['01_segmentations','02_bm_maps','03_measurements','04_background_predictions','05_background_measurements','Results','_object_predictions']}
    for d in dirs.values(): d.mkdir(parents=True, exist_ok=True)

    print(f'Found {len(imgs)} SumIP image(s). Output: {out}')
    for n, raw in enumerate(imgs,1):
        s = raw.stem
        print(f'\n===== [{n}/{len(imgs)}] {s} =====')
        seg = dirs['01_segmentations']/f'{s}_simple_segmentation.h5'
        if not (args.skip_existing and seg.exists()):
            run([ilastik,'--headless',f'--project={MODELS/"step1.ilp"}','--readonly','--export_source=Simple Segmentation','--output_format=compressed hdf5',f'--output_filename_format={seg}',str(raw)])

        bm_tif = dirs['02_bm_maps']/f'{s}_bm_map.tiff'; bm_h5 = dirs['02_bm_maps']/f'{s}_bm_map_h5ver.h5'
        if not (args.skip_existing and bm_h5.exists()): make_bm_map(seg,bm_tif,bm_h5,image_hw(raw))

        csv3 = dirs['03_measurements']/f'{s}.csv'; pred3 = dirs['_object_predictions']/f'{s}_bm_objects.h5'
        if not (args.skip_existing and csv3.exists()):
            run([ilastik,'--headless',f'--project={MODELS/"step3.ilp"}','--readonly',f'--table_filename={csv3}',f'--output_filename_format={pred3}','--output_format=compressed hdf5','--raw_data',str(raw),'--segmentation_image',f'{bm_h5}/bm_map_img'])

        bg = dirs['04_background_predictions']/f'{s}_probabilities.h5'
        if not (args.skip_existing and bg.exists()):
            run([ilastik,'--headless',f'--project={MODELS/"step4_pt1.ilp"}','--readonly','--export_source=Probabilities','--output_format=compressed hdf5',f'--output_filename_format={bg}',str(raw)])

        csv4 = dirs['05_background_measurements']/f'{s}.csv'; pred4 = dirs['_object_predictions']/f'{s}_background_objects.h5'
        if not (args.skip_existing and csv4.exists()):
            run([ilastik,'--headless',f'--project={MODELS/"step4_pt2.ilp"}','--readonly',f'--table_filename={csv4}',f'--output_filename_format={pred4}','--output_format=compressed hdf5','--raw_data',str(raw),'--prediction_maps',f'{bg}/exported_data'])

    final = dirs['Results']/'analysis_results.xlsx'
    bm_df,_ = summarize_csv_folder(dirs['03_measurements'],'BM','BM')
    ibg_df,_ = summarize_csv_folder(dirs['05_background_measurements'],'IBG','IBG')
    with pd.ExcelWriter(final, engine='openpyxl') as xw:
        bm_df.to_excel(xw, sheet_name='BM', index=False)
        ibg_df.to_excel(xw, sheet_name='IBG', index=False)
    print('\nANALYSIS COMPLETE')
    print(final)

if __name__ == '__main__':
    main()
