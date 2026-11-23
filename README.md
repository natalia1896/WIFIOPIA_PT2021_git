# WIFIOPIA / OISI analysis pipeline

Python and Jupyter workflow for wide-field optical intrinsic signal imaging (OISI): image registration, hemodynamic signal processing, anatomical and functional ROIs, and functional connectivity. Additional modules cover behavior/REST selection, spontaneous activity, hemorrhage, and lesion assessment.

**This repository contains analysis code and the study configuration, but not experimental recordings or a ready-to-run dataset.** Manual quality-control steps and study-specific input data are required.

## Installation

See [INSTALL_ANACONDA.md](INSTALL_ANACONDA.md) for a short Windows/macOS setup guide. Dependencies are listed in [environment.yml](environment.yml); the version ranges are adapted from the author's recent environment and **have not been validated end to end on both platforms**. Install [Ilastik](https://www.ilastik.org/) separately for Step 1–2 segmentation.

## Analysis workflow

Follow the order below. Run the indicated notebook or Python entry point; manual review is required where noted.

| Order | Stage | Entry point | Main purpose |
|---|---|---|---|
| 1 | Project creation | [Step1_ProjectCreation.ipynb](notebook/Step1_ProjectCreation.ipynb) | Discover recordings, create a manifest and reference BMPs |
| 1a | Segmentation (manual) | Ilastik | Produce HDF5 probability maps from reference images |
| 2 | Masks and registration | [Step2_MasksCreation.ipynb](notebook/Step2_MasksCreation.ipynb) | Build masks, register recordings and review bregma |
| 3 | Anatomical review (manual) | [run_step3_all_groups.py](src/run_step3_all_groups.py) | Review registration and anatomical landmarks in Napari |
| 4 | Hemodynamic signals | [Step4_SignalProcessing.ipynb](notebook/Step4_SignalProcessing.ipynb) | Convert BMP recordings to raw Zarr and processed Hb signals |
| 5 | Stimulation responses (manual) | [step5_response_detection.py](src/step5_response_detection.py) | Review and save functional response ROIs where applicable |
| 6 | Allen atlas registration (manual) | [Step6_AtlasTransformation.ipynb](notebook/Step6_AtlasTransformation.ipynb) | Align anatomical landmarks and atlas regions |
| Before 6b | Behavior / REST QC | [Behavior and REST workflow](#behavior-and-rest-qc) | Prepare accepted REST windows and shadow masks |
| Before 6b | Hemorrhage review, where required | [hemorrhage_mask_gui.py](src/hemorrhage_mask_gui.py) | Manually annotate hemorrhage masks |
| 6b | FC geometry and ROI eligibility | [Step6b_FC_Geometry_Coverage.ipynb](notebook/Step6b_FC_Geometry_Coverage.ipynb) | Build ROI geometry, coverage and eligibility tables |
| 7 | Functional connectivity | [Step7_FunctionalConnectivity.ipynb](notebook/Step7_FunctionalConnectivity.ipynb) | Compute connectivity maps and summary tables |

Behavior/REST review can start once Step4 signals are available; it must be completed **before Step6b** if the selected analysis requires its products. The `with_rest` branch uses accepted REST windows for Awake recordings, whereas `without_rest` uses full-recording windows. Keep the selected REST branch consistent across Step6b, Step7, and any comparisons to spontaneous activity. The full Step6b geometry workflow also expects the authoritative REST shadow-mask table, including for `without_rest`.

### Behavior and REST QC

Use [behavior_review_gui_step4.py](src/behavior_review_gui_step4.py) to annotate behavior, [rest_segment_review_gui.py](src/rest_segment_review_gui.py) to review candidate REST windows, then [REST_Finalizer_and_Behavior_Statistics.ipynb](notebook/REST_Finalizer_and_Behavior_Statistics.ipynb) to produce final REST segments and behavioral summaries. These data are inputs to analyses using the `with_rest` branch.

### Additional analyses

| Analysis | Entry points |
|---|---|
| Spontaneous hemodynamics | [Step_SpontaneousHemodynamics.ipynb](notebook/Step_SpontaneousHemodynamics.ipynb); optional [Napari signal review](src/spontaneous_signal_review_gui.py) |
| Lesion assessment | [damage_assessment_manual_gui.py](src/damage_assessment_manual_gui.py), [postprocessing](notebook/DamageAssessment_Postprocess_summary.ipynb), [atlas overlap](notebook/Damage_Lesion_Canonical_Overlap.ipynb) |
| Anesthesia onset | [run_step1_anest_start.py](src/run_step1_anest_start.py), [AnestStart_Step4_Hb_Analysis.ipynb](notebook/AnestStart_Step4_Hb_Analysis.ipynb) |
| Add new recording days | [Step1_AddExpDay.ipynb](notebook/Step1_AddExpDay.ipynb), [Step2_AddExpDay.ipynb](notebook/Step2_AddExpDay.ipynb); optional [re-registration script](src/step2_reregister_stems.py) |

## Data and configuration

The repository includes the **original study configuration** `config_15102026.yaml` and two calibration CSV files used by Step4:

- `Ext_hb_coeff.csv` — wavelength-dependent hemoglobin extinction coefficients; configured by `signal_processing.hemoglobin.coeff_abs_path`.
- `Est_pathlenght.csv` — estimated optical pathlengths; configured by `signal_processing.hemoglobin.pathlength_path` (the original filename spelling is preserved).

These files document the inputs and parameters used for the accompanying study. The original YAML points to the CSV files at `H:\Ext_hb_coeff.csv` and `H:\Est_pathlenght.csv`; placing the files in the repository does **not** automatically update those absolute paths. To rerun the analysis, **copy** `config_15102026.yaml` to a local working configuration (for example, `config.yaml`) and update the recording, output, and calibration paths for your computer. The scientific parameters in the published config remain unchanged.

Experimental recordings, Ilastik-generated HDF5 files, analysis results, and downloaded Allen atlas data are **not included**. Some notebooks also retain absolute paths from the original analysis: update them **in your local working copies** before execution. Start Jupyter from `notebook/` so that the adjacent `src/` directory can be found.

The notebooks are intended for publication **with saved cell outputs and execution counts cleared**. This repository does not include the experimental figures and numerical results generated during the study; run the notebooks with your own data to generate new outputs. Before running any cell, review flags such as `COMMIT`, `RUN_BATCH`, `OVERWRITE`, and `SAVE_*`, which may write or replace results.

## Hemodynamic method and reference

In Step4, changes in oxyhemoglobin (HbO), deoxyhemoglobin (HHb), and total hemoglobin (HbT) are estimated from dual-wavelength reflectance using a modified Beer–Lambert framework **following Ma et al. (2016)**. The implemented inversion uses wavelength-dependent extinction coefficients and optical pathlengths read from configured external CSV tables; acquisition-specific choices are not assumed to be identical to every example in the paper.

**Ma, Y., et al. (2016).** Wide-field optical mapping of neural activity and brain haemodynamics: considerations and novel approaches. *Philosophical Transactions of the Royal Society B*, **371**, 20150360. [https://doi.org/10.1098/rstb.2015.0360](https://doi.org/10.1098/rstb.2015.0360).
