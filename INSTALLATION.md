# Installation with Conda (Windows and macOS)

This is a research workflow, not an installable Python package. Experimental recordings and Allen atlas data are not included; the original study configuration and two hemoglobin calibration CSV files are provided.

## 1. Get Conda and the project

Install [Anaconda](https://www.anaconda.com/download) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/main). Download the repository ZIP from GitHub and extract it, or clone the repository with Git.

Open **Anaconda Prompt** (Windows) or **Terminal** (macOS). Navigate to the extracted project folder containing `environment.yml`, `src/`, and `notebook/`.

## 2. Create the environment

```bash
conda env create -f environment.yml
conda activate md-analysis
```

The supplied [`environment.yml`](environment.yml) uses Python 3.11, conda-forge, and version ranges adapted from the author's recent environment. It has **not** been fully solved and run on every operating system. Zarr 3 is selected; Step4 requests **Zarr storage format 2** where supported. Verify the environment with your data before relying on it for a full analysis.

Optionally register a Jupyter kernel:

```bash
python -m ipykernel install --user --name md-analysis --display-name "OISI (md-analysis)"
```

## 3. Configure your data

Copy `config_15102026.yaml` to a working `config.yaml`, leaving the published study config unchanged. Update the raw-data and output paths and the original `H:\` paths to `Ext_hb_coeff.csv` and `Est_pathlenght.csv` under `signal_processing.hemoglobin`. The calibration CSV files are supplied in the repository root, but their YAML paths must still point to the actual files on your computer.

**Important:** some notebooks contain historical absolute paths. Update `CONFIG_PATH` and any other file paths **in your local notebook copies** before running them; changing `config.yaml` alone does not update hard-coded notebook paths.

Install [Ilastik](https://www.ilastik.org/) separately to produce the HDF5 probability maps required by Step2. BrainGlobe may download the selected Allen atlas when first used. Video workflows may use `ffmpeg`/`ffprobe` supplied by the environment.

## 4. Start Jupyter Notebook or JupyterLab

From the project root, start **Jupyter Notebook**:

```bash
cd notebook
jupyter notebook
```

Alternatively, you can use **JupyterLab** from the same `notebook/` directory:

```bash
jupyter lab
```

Select the environment's Jupyter kernel and follow the stage order in [README.md](README.md). Start either interface from `notebook/` because some notebooks resolve `src/` relative to the kernel working directory.

For a CLI entry point, return to the project root and run, for example:

```bash
python src/step4_signal_processing.py --help
```

Before executing processing cells, check `COMMIT`, `RUN_BATCH`, `OVERWRITE`, and `SAVE_*` settings. Published notebooks are intended to have cleared cell outputs; computation requires your own data and configuration.
