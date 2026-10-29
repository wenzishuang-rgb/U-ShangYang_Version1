# U-ShangYang

Python scripts for computing moisture-transport fields from ERA5, training the U-ShangYang reconstruction model, and generating atmospheric-river candidate masks.

This repository contains three core scripts. `compute_ivt.py` converts ERA5 specific humidity and winds into monthly IVT fields. `train_model.py` trains the model and saves checkpoints. `postprocess.py` loads a checkpoint, performs inference, and applies segmentation and whole-object filters independently at each requested analysis time.

## Files

| File | Purpose |
| --- | --- |
| `compute_ivt.py` | ERA5 dimension handling, pressure-weighted transport calculation and monthly IVT NetCDF output. |
| `train_model.py` | Model definition, masked-reconstruction training, validation and checkpoint saving. |
| `postprocess.py` | Model inference, PCA, Otsu segmentation, object filtering and diagnostic output. |
| `requirements.txt` | Direct runtime dependencies; original package versions were not supplied. |
| `.gitignore` | Excludes local environments, credentials, datasets and model checkpoints from Git tracking. |
| `LICENSE` | MIT License copied from the existing project repository. |

## Scope

The workflow starts with ERA5 pressure-level specific humidity (`q`) and winds (`u`, `v`). `compute_ivt.py` prepares the `IVT`, `uIVT` and `vIVT` inputs used by both subsequent scripts. The supplied pressure-weighting convention is preserved and documented below.

The postprocessing script writes one diagnostic NetCDF file per requested analysis time. Its final variables are `final_ar_candidate_mask` and `final_ar_candidate_object_id`. These are not yet the annual files with a `time` dimension and variables named `ar_mask` and `ar_object_id` described for the released dataset. Annual AR-label assembly, variable renaming, conversion of the mask to `uint8`, leap-day exclusion, and the full 1980–2024 label-processing schedule are not implemented in these scripts.

Model weights and example atmospheric input files are not bundled. For reproduction of an existing dataset release, use its actual production checkpoint and parameter settings. Retraining a model alone does not establish bitwise equivalence to that release.

## Installation

Create a Python environment and install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1` instead. Select a PyTorch installation suitable for the available CPU or CUDA environment.

Cartopy is optional. Install it separately if coastlines and map projections are required:

```bash
python -m pip install cartopy
```

Without Cartopy, the script uses ordinary longitude–latitude axes. Cartopy may require its coastline data to be available on the machine.

The supplied sources did not specify the original Python, PyTorch, CUDA or package versions. `requirements.txt` is therefore a dependency list, not a locked reproduction environment. Record the actual working versions before creating a permanent software release.

## Input files

### ERA5 pressure-level inputs

`compute_ivt.py` searches for raw files matching:

```text
RAW_DATA_ROOT/YYYY/data-YYYY-MM-*.nc
```

Each file must contain `q` in kg kg−1 and `u`, `v` in m s−1. Pressure coordinates must be in **hPa**. The script accepts `pressure_level` or `level`, and `valid_time` or `time`, as dimension names. Latitude and longitude must be named `latitude` and `longitude`.

The variable arrays must be ordered `(time, pressure level, latitude, longitude)`, using the accepted aliases above. The pressure reduction uses `axis=1`; the script does not reorder arbitrary dimension layouts. It selects all supplied pressure levels from 200 through 1000 hPa and sorts them in descending order. The number of supplied levels is not enforced.

Input data are expected to have the intended 0.25° grid and six-hourly analysis times. This script does not regrid or resample the source data. Files within a month must represent the same physical grid: their coordinates are reassigned to the first file's grid before concatenation. Duplicate or missing time records are not automatically corrected, and failed input files are reported and skipped.

### Monthly IVT inputs to training and postprocessing

Use monthly NetCDF files under one subdirectory per year, for example:

```text
/path/to/IVT/2016/IVT_2016_06.nc
```

Each file must contain `IVT`, `uIVT` and `vIVT` on a shared global latitude–longitude grid. The training loader expects the variable arrays in `(time, latitude, longitude)` order. The postprocessor requires `time`, `latitude` and `longitude` coordinates; it also accepts `lat` and `lon` dimension names and renames them internally.

Use IVT and its vector components in the same units and preprocessing convention as the training inputs. No additional standardization is performed by these scripts. The postprocessor matches the requested time to the file at minute precision and requires exactly one match. Longitude connectivity is periodic, so the algorithms are intended for a global grid.

Training discovers `.nc` filenames containing `IVT` within each year directory. Postprocessing uses the exact filename pattern `IVT_YYYY_MM.nc`.

## 1. Compute monthly IVT fields

Edit the path constants in `compute_ivt.py`:

```python
RAW_DATA_ROOT = "/path/to/ERA5_quv"
IVT_OUT_ROOT = "/path/to/IVT"
```

Then run:

```bash
python compute_ivt.py
```

The configured period is 1980–2024. Monthly outputs use `IVT_OUT_ROOT/YYYY/IVT_YYYY_MM.nc`, with variables `uIVT`, `vIVT` and `IVT` in kg m−1 s−1. The transport variables are stored as `float32`, latitude and longitude as `float64`, and times are sorted. Successful inputs are combined even if other input files failed, so a monthly file's existence does not establish complete six-hourly coverage. All successfully read times are retained, including 29 February where present.

Set **`IVT_DIR` in `train_model.py` and `--data-root` in `postprocess.py` to the same directory as `IVT_OUT_ROOT`**. The supplied machine-specific defaults were preserved: IVT calculation currently writes to `/home/Wenzishuang/Wen_AR_0.25degree/IVT`, whereas the other two scripts default to `/Global/Wen_AR_0.25degree/IVT`. Align these settings before running the workflow.

### Pressure-weighting convention retained from the supplied code

For descending pressure coordinates, the script multiplies each `q*u` or `q*v` sample by the difference to the next pressure level, converted from hPa to Pa, and appends an additional **5000 Pa weight for the final selected level**. The weighted sum is divided by standard gravity, and `IVT` is the magnitude of its two components.

When both 1000 and 200 hPa are present, the consecutive differences sum to 80000 Pa, but the appended term makes the total weight **85000 Pa**. Thus this weighting is not an exact discretization over only the stated 80000 Pa pressure span. The final-layer treatment needs to be reconciled with the Methods description and the calculation actually used for the released data. It has not been silently replaced with a different quadrature rule, because doing so would alter the existing IVT inputs and downstream results.

## 2. Train the model

Edit the two path constants near the start of `train_model.py`. The supplied local paths have been preserved; replace them for your environment, for example:

```python
IVT_DIR = "/path/to/IVT"
OUTPUT = "./outputs/model"
```

Then run:

```bash
python train_model.py
```

The supplied training settings are retained:

| Setting | Value or behavior |
| --- | --- |
| Training years | 1980–2010 |
| Validation years | 2011–2015 |
| Monthly sampling | One randomly selected analysis per training file per access; one fixed sampled analysis per validation file. |
| Random seed | 42; validation time sampling uses 43. |
| Model | Three 3 × 3 convolutional encoder layers with 64 channels and ReLU, followed by a three-channel convolutional decoder. |
| Input masking | Independent element masking with probability 0.3 during training. |
| Objective | Mean squared reconstruction error over all input elements. |
| Validation | Reconstruction of unmasked validation inputs. |
| Optimizer | Adam with learning rate 0.0001. |
| Epochs | 120 |
| Batch sizes | Training: 4; validation: 2. |

The script selects CUDA if available and otherwise selects the CPU. It retains the original CUDA automatic-mixed-precision calls. Each data-loader cache can hold two complete monthly files, so sufficient host memory is needed for the actual input grid and time coverage.

Outputs under `OUTPUT` include:

- `checkpoints/best_model.pth`: model weights saved when validation loss improves.
- `checkpoints/last_training_state.pth`: the latest training state.
- `checkpoints/epoch_XXX.pth`: a training state for each completed epoch.
- `metadata/validation_random_time_indices.json`: sampled validation indices and period metadata.

An existing `last_training_state.pth` triggers automatic resumption. The original resume behavior is preserved: its saved validation loss initializes `best_loss`, and random-number-generator and gradient-scaler states are not restored. Use a separate output directory for a fresh training run.

`TEST_START = 2016` and `TEST_END = 2020` are written to metadata only. This script does not evaluate a test dataset. These metadata years do not define the catalogue-comparison period in the manuscript.

## 3. Generate candidate labels

After training, or after obtaining the actual trained checkpoint, run the postprocessor. The example below uses the **author-provided execution settings**: local Otsu with radius 48, joint global–local rank selection, and a final area filter that retains the largest class. The original script filename has been replaced by `postprocess.py`; input, checkpoint and output paths are shown explicitly for portability.

```bash
python postprocess.py \
  --data-root /path/to/IVT \
  --model-path ./outputs/model/checkpoints/best_model.pth \
  --output-root ./outputs/global_local_r48_area_keep_large \
  --time 2016-08-28T12:00:00 \
  --otsu-mode local \
  --local-otsu-radius 48 \
  --object-filter global-local-rank-multiotsu \
  --area-postfilter multiotsu-keep-large \
  --gpu 0
```

For a CPU run, replace `--gpu 0` with `--cpu`. On Windows, the command can be entered on one line. Repeat `--time` to process several analysis times. If no time is supplied, the original default is `2016-06-15T12:00:00`.

Inspect available options with:

```bash
python postprocess.py --help
```

The two scripts use matching encoder and decoder definitions. The training script's `best_model.pth` is a plain state dictionary; the postprocessor can also extract model weights from supported wrapped checkpoints.

### Selecting the object filter

| `--object-filter` | Behavior |
| --- | --- |
| `integrated-anomaly-multiotsu` | Original default. Retains the highest class of area-integrated local-anomaly scores; does not use global IVT ranks. |
| `global-local-rank-multiotsu` | Combines ranks of local integrated anomalies and global IVT support, then retains the highest joint-score class. |
| `area-multiotsu` | Selects complete objects using three-class Otsu on log10 physical area. |
| `none` | Retains all first-Otsu candidate objects before subsequent filters. |

The author-provided command explicitly selects the second option and then applies the area postfilter:

```text
--object-filter global-local-rank-multiotsu
--area-postfilter multiotsu-keep-large
```

The parser defaults remain unchanged from the supplied source, so keep these explicit arguments in the run command. Omitting them selects `integrated-anomaly-multiotsu` and disables the area postfilter.

The author-provided command leaves the tropical filter enabled: it removes a complete object when more than 50% of its area is between −15° and 15°. The sequence is therefore local Otsu segmentation, global–local rank selection, tropical screening, and the final physical-area filter. `--disable-tropical-filter` disables the tropical step. The other area-filter choices are `none` and `multiotsu-remove-small`.

The example establishes the supplied single-time configuration. Reproduction of the full dataset additionally requires the matching checkpoint, runtime environment, full time schedule and annual export workflow.

Keep `--otsu-mode local` for the documented local-anomaly workflow. The original global comparison branch is retained but needs separate scientific validation: it returns thresholds in raw PC1 units while the downstream local-excess calculation uses the scaled PC1 field. This formatting revision does not change that behavior.

### Outputs

| Output directory | Contents |
| --- | --- |
| `data/` | Per-time diagnostic NetCDF files and final-mask NumPy arrays. |
| `objects/` | Final object-property CSV files. |
| `diagnostics/` | JSON summaries and CSV records for the filtering stages. |
| `figures/` | Diagnostic figures. |

The NetCDF final-mask variable is `final_ar_candidate_mask` (`int8`, 0 or 1), and the object variable is `final_ar_candidate_object_id` (`int32`, 0 for background and positive integers for objects). Both have `(latitude, longitude)` dimensions. Time is recorded in the `matched_time` attribute. Object identifiers apply only within the current analysis and are not persistent tracks.

Use different output directories for different Otsu radii, checkpoints or tropical-filter settings. Filenames encode time and filter names, but not every configuration parameter, so reusing one output directory can overwrite a previous run with the same filename.

## Code preparation and validation

This version standardizes spacing, indentation, line wrapping and comments. Chinese comments, docstrings, command-line help and user-facing messages have been translated into English. Network layers, numerical calculations, file-variable names, random seeds, year ranges and filter defaults are preserved. The NetCDF `model` attribute now records the supplied checkpoint filename instead of the original hard-coded `best_model_81.pth` label; this changes metadata only.

Syntax, formatting and source-structure equivalence were checked with Python 3.12.14. A small synthetic IVT test also checks both supported dimension-name conventions, pressure sorting, agreement with the original calculation, monthly time sorting, output names and NetCDF storage types. This test preserves and explicitly checks the supplied 85000 Pa total weighting; it does not validate that weighting as an exact 1000–200 hPa integral.

These checks do not establish end-to-end scientific reproduction. Production ERA5 files, trained weights and the original runtime environment were not included, so full-period IVT processing, model training, GPU execution and production-label reproduction have not been run as part of this preparation.

## Contact

Thank you for your interest in U-ShangYang. If you have any questions, encounter issues with the code, or would like to discuss its use or potential collaboration, please feel free to contact me. Feedback and suggestions are always welcome.

**Zishuang Wen**

Email: [wenzishuang@stu.ouc.edu.cn](mailto:wenzishuang@stu.ouc.edu.cn)

State Key Laboratory of Physical Oceanography / College of Oceanic and Atmospheric Sciences / Center for Ocean Carbon Neutrality, Ocean University of China, Qingdao, China.

## License

This repository uses the MIT License. See the repository's [LICENSE](LICENSE) file for the full terms.
