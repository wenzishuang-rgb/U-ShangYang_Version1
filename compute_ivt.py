#!/usr/bin/env python3
"""Compute monthly IVT components from six-hourly ERA5 humidity and winds.

Edit RAW_DATA_ROOT and IVT_OUT_ROOT before running. This script preserves the
supplied discrete pressure-weighting rule, including its terminal 5000 Pa term.
See README.md for the associated boundary-treatment note.

For questions, feedback or collaboration, please contact Zishuang Wen at
wenzishuang@stu.ouc.edu.cn. See README.md for affiliation and usage details.
"""

import xarray as xr
import numpy as np
import os
import glob
import warnings
from scipy import constants
from pathlib import Path
import pandas as pd

RAW_DATA_ROOT = "/Data025/Data-for-quv-6hourly"
IVT_OUT_ROOT = "/home/Wenzishuang/Wen_AR_0.25degree/IVT"
START_YEAR = 1980
END_YEAR = 2024
IVT_PLEV_MIN = 200
IVT_PLEV_MAX = 1000
g = constants.g
warnings.filterwarnings("ignore")


def get_era5_dims(ds):
    """Normalize ERA5 pressure-level and time dimension names."""
    lev_dim = "pressure_level" if "pressure_level" in ds.dims else "level"
    time_dim = "valid_time" if "valid_time" in ds.dims else "time"
    if lev_dim != "level":
        ds = ds.rename({lev_dim: "level"})
    if time_dim != "time":
        ds = ds.rename({time_dim: "time"})
    return ds


def create_year_folder(year):
    """Create the output directory for one year."""
    year_folder = Path(f"{IVT_OUT_ROOT}/{year}")
    year_folder.mkdir(parents=True, exist_ok=True)
    return year_folder


def get_sorted_monthly_files(year, month):
    """Find input NetCDF files for the requested year and month."""
    mon_str = str(month).zfill(2)
    file_pattern = f"{RAW_DATA_ROOT}/{year}/data-{year}-{mon_str}-*.nc"
    file_list = sorted(glob.glob(file_pattern))
    if not file_list:
        print(
            f"Warning: {year}-{mon_str}: no input files found | Pattern: {file_pattern}"
        )
    else:
        print(f"{year}-{mon_str}: found {len(file_list)} six-hourly input files")
    return file_list


def calculate_single_ivt(file_path):
    """Compute transport using the supplied pressure-weighting convention.

    Input fields must have time, pressure, latitude and longitude axes in
    that order. Pressure coordinates are assumed to be in hPa.
    """
    try:
        ds = xr.open_dataset(
            file_path, engine="netcdf4", mask_and_scale=True, decode_times=True
        )
        ds = get_era5_dims(ds)
        q = ds["q"]
        u = ds["u"]
        v = ds["v"]
        plevs = ds["level"].values

        plev_mask = (plevs >= IVT_PLEV_MIN) & (plevs <= IVT_PLEV_MAX)
        plevs_selected = plevs[plev_mask]
        plevs_sorted = np.sort(plevs_selected)[::-1]

        q = q.sel(level=plevs_sorted)
        u = u.sel(level=plevs_sorted)
        v = v.sel(level=plevs_sorted)

        delp = [
            (plevs_sorted[i + 1] - plevs_sorted[i]) * -100
            for i in range(len(plevs_sorted) - 1)
        ]
        # Retain the supplied terminal 5000 Pa weight for numerical consistency.
        # See the pressure-weighting note in README.md before changing this rule.
        delp.append(5000.0)
        delp = np.array(delp)[:, np.newaxis, np.newaxis]

        uIVT = np.nansum(q * u * delp, axis=1) / g
        vIVT = np.nansum(q * v * delp, axis=1) / g
        IVT = np.sqrt(uIVT**2 + vIVT**2)

        ivt_result = xr.Dataset(
            data_vars={
                "uIVT": (
                    ("time", "latitude", "longitude"),
                    uIVT,
                    {
                        "units": "kg m-1 s-1",
                        "long_name": "Zonal Integrated Water Vapor Transport (1000-200hPa)",
                        "description": "Eastward positive, vertical integral of q*u*dp/g",
                    },
                ),
                "vIVT": (
                    ("time", "latitude", "longitude"),
                    vIVT,
                    {
                        "units": "kg m-1 s-1",
                        "long_name": "Meridional Integrated Water Vapor Transport (1000-200hPa)",
                        "description": "Northward positive, vertical integral of q*v*dp/g",
                    },
                ),
                "IVT": (
                    ("time", "latitude", "longitude"),
                    IVT,
                    {
                        "units": "kg m-1 s-1",
                        "long_name": "Integrated Water Vapor Transport (1000-200hPa)",
                        "description": "Vector magnitude of uIVT and vIVT, core index for AR detection",
                    },
                ),
            },
            # Preserve coordinate precision with float64 latitude and longitude.
            coords={
                "time": ds["time"],
                "latitude": ds["latitude"].astype(np.float64),
                "longitude": ds["longitude"].astype(np.float64),
            },
            attrs={
                "data_source": "ERA5 0.25deg 6-hourly q/u/v data",
                "calculation_range": "1000hPa to 200hPa vertical integral",
                "gravity_acceleration": f"{g:.6f} m s-2",
                "original_file": os.path.basename(file_path),
                "processing_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        ivt_result.latitude.attrs = ds.latitude.attrs
        ivt_result.longitude.attrs = ds.longitude.attrs
        ds.close()
        return ivt_result

    except Exception as e:
        print(
            f"IVT calculation failed: {os.path.basename(file_path)} | Error: {str(e)[:80]}"
        )
        return None


# Monthly processing and NetCDF output
def batch_calc_era5_ivt():
    """Compute and save monthly IVT fields over the configured years."""
    for year in range(START_YEAR, END_YEAR + 1):
        year_folder = create_year_folder(year)
        print(f"\n=====================================")
        print(f"Processing {year} ERA5 input data")
        print(f"=====================================")

        for month in range(1, 13):
            mon_str = str(month).zfill(2)
            print(f"\nProcessing month {year}-{mon_str}")
            monthly_files = get_sorted_monthly_files(year, month)
            if not monthly_files:
                continue

            monthly_ivt_results = []
            total_files = len(monthly_files)
            for idx, file_path in enumerate(monthly_files, 1):
                ivt_ds = calculate_single_ivt(file_path)
                if ivt_ds is not None:
                    monthly_ivt_results.append(ivt_ds)
                if idx % 20 == 0 or idx == total_files:
                    print(
                        f"  Progress: {idx}/{total_files} | Successful IVT results: {len(monthly_ivt_results)}"
                    )

            if monthly_ivt_results:
                try:
                    # Use the first input grid as the common monthly coordinate grid.
                    standard_lat = monthly_ivt_results[0]["latitude"]
                    standard_lon = monthly_ivt_results[0]["longitude"]
                    for i in range(len(monthly_ivt_results)):
                        monthly_ivt_results[i] = monthly_ivt_results[i].assign_coords(
                            latitude=standard_lat, longitude=standard_lon
                        )
                    # Concatenate analyses and sort by time.
                    ivt_monthly = xr.concat(monthly_ivt_results, dim="time").sortby(
                        "time"
                    )
                    out_file_name = f"IVT_{year}_{mon_str}.nc"
                    out_file_path = year_folder / out_file_name
                    # Store transport fields as float32 and coordinates as float64.
                    encoding_config = {
                        "uIVT": {"dtype": "float32", "zlib": True, "complevel": 5},
                        "vIVT": {"dtype": "float32", "zlib": True, "complevel": 5},
                        "IVT": {"dtype": "float32", "zlib": True, "complevel": 5},
                        "latitude": {
                            "dtype": np.float64
                        },  # Preserve latitude precision.
                        "longitude": {
                            "dtype": np.float64
                        },  # Preserve longitude precision.
                        "time": {"units": "hours since 1900-01-01"},
                    }
                    # Write the monthly NetCDF file.
                    ivt_monthly.to_netcdf(
                        out_file_path,
                        encoding=encoding_config,
                        engine="netcdf4",
                        format="NETCDF4",
                    )
                    ivt_monthly.close()
                    monthly_ivt_results.clear()

                    print(f"Saved monthly IVT file: {out_file_path}")
                    print(
                        f"Output: {len(ivt_monthly.time)} analyses | Input grid: 0.25 x 0.25 degrees | Variables: uIVT/vIVT/IVT"
                    )

                except Exception as e:
                    print(
                        f"Error: {year}-{mon_str}: monthly concatenation or saving failed | Error: {str(e)[:80]}"
                    )
                    monthly_ivt_results.clear()
                    continue
            else:
                print(
                    f"Warning: {year}-{mon_str}: no successful IVT results; no monthly file was written"
                )

    print(f"\n=====================================")
    print(f"ERA5 IVT batch processing complete.")
    print(f"Output root: {IVT_OUT_ROOT}")
    print(
        f"Output layout: {IVT_OUT_ROOT}/YYYY/IVT_YYYY_MM.nc (up to 12 monthly files per year)"
    )
    print(f"Variables: uIVT (zonal), vIVT (meridional), IVT (vector magnitude)")
    print(f"Selected pressure levels: 1000 to 200 hPa")
    print(f"Transport units: kg m-1 s-1")
    print(f"=====================================")


# Script entry point
if __name__ == "__main__":
    batch_calc_era5_ivt()
