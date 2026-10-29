# U-ShangYang

Code for preparing ERA5 moisture-transport fields, training the U-ShangYang model, and generating atmospheric-river labels.

| Script | Purpose |
| --- | --- |
| `compute_ivt.py` | Calculate monthly IVT fields from ERA5 humidity and winds. |
| `train_model.py` | Train the model and save its weights. |
| `postprocess.py` | Load model weights and generate AR masks and object identifiers. |

## Installation

Run the following commands from the repository directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with `.venv\Scripts\Activate.ps1`. Cartopy is optional and can be installed with `python -m pip install cartopy` for map projections and coastlines.

## 1. Calculate IVT

In `compute_ivt.py`, set your input and output directories:

```python
RAW_DATA_ROOT = "/path/to/ERA5_quv"
IVT_OUT_ROOT = "/path/to/IVT"
```

Arrange the six-hourly ERA5 files as `RAW_DATA_ROOT/YYYY/data-YYYY-MM-*.nc`. Each file must contain `q`, `u` and `v`, ordered by time, pressure level, latitude and longitude. Pressure coordinates must use hPa. The script accepts `time` or `valid_time`, and `level` or `pressure_level`; spatial coordinates must be named `latitude` and `longitude`.

```bash
python compute_ivt.py
```

Monthly files are saved as `IVT_OUT_ROOT/YYYY/IVT_YYYY_MM.nc`, containing `IVT`, `uIVT` and `vIVT`. Adjust `START_YEAR` and `END_YEAR` in the script if needed.

## 2. Train the model

In `train_model.py`, set `IVT_DIR` to the same directory as `IVT_OUT_ROOT`:

```python
IVT_DIR = "/path/to/IVT"
OUTPUT = "./outputs/model"
```

The default training period is 1980–2010, with validation during 2011–2015. Adjust the year constants in the script if needed.

```bash
python train_model.py
```

The selected model weights are saved to `./outputs/model/checkpoints/best_model.pth`. Existing training checkpoints are resumed automatically; use a new output directory for a fresh run.

## 3. Generate AR labels

Use the same IVT directory and the trained checkpoint:

```bash
python postprocess.py \
  --data-root /path/to/IVT \
  --model-path ./outputs/model/checkpoints/best_model.pth \
  --output-root ./outputs/labels \
  --time 2016-08-28T12:00:00 \
  --gpu 0 \
  --otsu-mode local \
  --local-otsu-radius 48 \
  --object-filter global-local-rank-multiotsu \
  --area-postfilter multiotsu-keep-large
```

Replace `--gpu 0` with `--cpu` for CPU execution. Repeat `--time` to process additional analysis times. On Windows, enter the command on one line. To list all options, run `python postprocess.py --help`.

NetCDF files are saved under `./outputs/labels/data/`. Object tables, diagnostic records and figures are saved in the `objects/`, `diagnostics/` and `figures/` subdirectories, respectively.

## Output variable names

The final variables in the postprocessing output correspond to the released dataset variables as follows:

| Postprocessing output | Released dataset | Meaning |
| --- | --- | --- |
| `final_ar_candidate_mask` | `ar_mask` | Binary AR mask: 1 = AR; 0 = non-AR. |
| `final_ar_candidate_object_id` | `ar_object_id` | Positive integers identify individual AR objects; 0 = non-AR. |

Object identifiers apply within each analysis time and do not represent tracks across successive analyses. The postprocessing script retains the output names shown in the first column.

## Contact

Thank you for your interest in U-ShangYang. If you have any questions, encounter issues with the code, or would like to discuss its use or potential collaboration, please feel free to contact me. Feedback and suggestions are always welcome.

**Zishuang Wen**

Email: [wenzishuang@stu.ouc.edu.cn](mailto:wenzishuang@stu.ouc.edu.cn)

State Key Laboratory of Physical Oceanography / College of Oceanic and Atmospheric Sciences / Center for Ocean Carbon Neutrality, Ocean University of China, Qingdao, China.

## License

This project uses the [MIT License](LICENSE). Additional implementation details are available in [Technical notes](TECHNICAL_NOTES.md).
