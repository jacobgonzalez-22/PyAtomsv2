# -*- coding: utf-8 -*-
"""
PyAtoms Experimental Data Loader

Created on Sep 23 2026
@author: Jacob Gonzalez

Modification Log
----------------
2026-09-23 - Jacob Gonzalez
    - Added experimental STM data support
    - Added Nanonis SXM loading through xarray-nanonis
    - Added channel and scan-direction selection support
    - Added centered physical-coordinate handling for comparison with PyAtoms simulation
    - Preserved seperate raw and processed experimental data arrays
"""

import os

import numpy as np
import xarray as xr

class ExperimentalData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

        self.dataset = xr.open_dataset(file_path, engine="nanonis")

        self.channels = list(self.dataset.data_vars)
        self.directions = [str(value) for value in self.dataset["dir"].values]

        self.channel = None
        self.direction = None

        self.raw = None
        self.processed = None

        self.leveling = "None"
        self.line_flattening = "None"

        self.x_nm = np.asarray(self.dataset["x"].values, dtype=float) * 1e9
        self.y_nm = np.asarray(self.dataset["y"].values, dtype=float) * 1e9

        # pyatoms uses coordinates centered around (0, 0)
        self.x_center_nm = (np.min(self.x_nm) + np.max(self.x_nm)) / 2
        self.y_center_nm = (np.min(self.y_nm) + np.max(self.y_nm)) / 2

        self.x_display_nm = self.x_nm - self.x_center_nm
        self.y_display_nm = self.y_nm - self.y_center_nm

        self.nx = len(self.x_nm)
        self.ny = len(self.y_nm)

        self.x_range_nm = float(np.max(self.x_display_nm) - np.min(self.x_display_nm))
        self.y_range_nm = float(np.max(self.y_display_nm) - np.min(self.y_display_nm))

        self.metadata = dict(self.dataset.attrs)


    def load_channel(self, channel="Z", direction="forward"):
        if channel not in self.channels:
            raise ValueError(f"Channel '{channel}' not found in dataset. Available channels: {self.channels}")

        if direction not in self.directions:
            raise ValueError(f"Direction '{direction}' not found in dataset. Available directions: {self.directions}")

        data = self.dataset[channel].sel(dir=direction)

        self.channel = channel
        self.direction = direction

        # load raw data
        self.raw = np.asarray(data.values, dtype=float).copy()

        # start from the unprocessed channel data
        self.processed = self.raw.copy()

        return self.processed

    def apply_processing(self, leveling="None", line_flattening="None"):
        if self.raw is None:
            return None

        # always restart from the original experimental data
        processed = np.asarray(self.raw, dtype=float).copy()

        # global leveling
        if leveling == "None":
            pass

        elif leveling == "Mean":
            finiteMask = np.isfinite(processed)

            if np.any(finiteMask):
                processed = processed - np.mean(processed[finiteMask])

        elif leveling == "Plane":
            X, Y = np.meshgrid(self.x_display_nm, self.y_display_nm)

            finiteMask = np.isfinite(processed)

            if np.count_nonzero(finiteMask) >= 3:
                A = np.column_stack(
                    (
                        X[finiteMask],
                        Y[finiteMask],
                        np.ones(np.count_nonzero(finiteMask))
                    )
                )

                coefficients, _, _, _ = np.linalg.lstsq(A, processed[finiteMask], rcond=None)

                a, b, c = coefficients
                plane = a * X + b * Y + c

                processed = processed - plane

        else:
            raise ValueError(f"Unknown leveling mode: {leveling}")

        # line-by-line flattening
        if line_flattening == "None":
            pass

        elif line_flattening == "Mean":
            for rowIndex in range(processed.shape[0]):
                row = processed[rowIndex, :]
                finiteMask = np.isfinite(row)

                if np.any(finiteMask):
                    rowMean = np.mean(row[finiteMask])
                    processed[rowIndex, finiteMask] = row[finiteMask] - rowMean

        elif line_flattening == "Median":
            for rowIndex in range(processed.shape[0]):
                row = processed[rowIndex, :]
                finiteMask = np.isfinite(row)

                if np.any(finiteMask):
                    rowMedian = np.median(row[finiteMask])
                    processed[rowIndex, finiteMask] = row[finiteMask] - rowMedian

        elif line_flattening == "Linear":
            x = self.x_display_nm

            for rowIndex in range(processed.shape[0]):
                row = processed[rowIndex, :]
                finiteMask = np.isfinite(row)

                if np.count_nonzero(finiteMask) < 2:
                    continue

                A = np.column_stack(
                    (
                        x[finiteMask],
                        np.ones(np.count_nonzero(finiteMask))
                    )
                )

                coefficients, _, _, _ = np.linalg.lstsq(A, row[finiteMask], rcond=None)

                slope, offset = coefficients
                lineBackground = slope * x + offset

                processed[rowIndex, finiteMask] = row[finiteMask] - lineBackground[finiteMask]

        else:
            raise ValueError(f"Unknown line flattening mode: {line_flattening}")

        self.leveling = leveling
        self.line_flattening = line_flattening
        self.processed = processed

        return self.processed

    def reset_processing(self):
        if self.raw is None:
            return

        self.processed = self.raw.copy()
        self.leveling = "None"
        self.line_flattening = "None"

    def get_extent(self):
        return [
            float(np.min(self.x_display_nm)),
            float(np.max(self.x_display_nm)),
            float(np.min(self.y_display_nm)),
            float(np.max(self.y_display_nm))
        ]

    