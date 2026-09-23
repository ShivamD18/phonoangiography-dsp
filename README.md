# Phonoangiography DSP

A reproducible research scaffold for ingesting, inspecting, and preprocessing vascular sound recordings.

## Setup

Using conda:

```powershell
conda env create -f environment.yml
conda activate phonoangiography-dsp
```

Using pip:

```powershell
python -m pip install -r requirements.txt
```

Place untouched WAV recordings in `data/raw/`. The project intentionally keeps raw, interim, and processed data out of git.

## Workflow

1. Put recordings in `data/raw/`.
2. Run `jupyter lab` and open `notebooks/01_data_ingestion_and_eda.ipynb`.
3. Use `src.utils.audio_io` for consistent loading and metadata inspection.
4. Record filter decisions and observations in `docs/`.

The package is importable from notebooks when the notebook is launched from the project root. The notebook also adds the project root to `sys.path` for predictable local execution.
