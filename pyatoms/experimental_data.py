# -*- coding: utf-8 -*-
"""
PyAtoms Experimental Data Loader

Created on Sep 23 2026
@author: Jacob Gonzalez

Modification Log
----------------
2026-10-01 - Jacob Gonzalez
    - Added Gwyddion GWY experimental data support
    - Added file-content detection so GWY files with an .sxm extension
      are still loaded correctly
    - Added NumPy compatibility workaround for xarray-nanonis SXM loading

2026-09-23 - Jacob Gonzalez
    - Added experimental STM data support
    - Added Nanonis SXM loading through xarray-nanonis
    - Added channel and scan-direction selection support
    - Added centered physical-coordinate handling for comparison with PyAtoms simulation
    - Preserved seperate raw and processed experimental data arrays
"""

import os
import re

import numpy as np

# compatibility fix for xarray-nanonis:
# xarray-nanonis still uses the deprecated np.long alias which seems to be unavailable in modern numpy versions
if "long" not in np.__dict__:
    np.long = np.int_

import xarray as xr
import gwyfile

class ExperimentalData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)

        self.dataset = None
        self.file_format = None

        # gwyddion specific storage
        self._gwy_container = None
        self._gwy_fields = {}
        self._gwy_field_ids = {}

        self.channels = []
        self.directions = []

        self.channel = None
        self.direction = None

        self.raw = None
        self.processed = None

        self.leveling = "None"
        self.line_flattening = "None"

        self.metadata = {}

        # detect by actual file contents instead of trusting the extension
        if self._is_gwyddion_file():
            self.file_format = "gwyddion"
            self._load_gwyddion()
        else:
            self.file_format = "nanonis"
            self._load_nanonis()

    def _is_gwyddion_file(self):
        """
        detects a gwyddion native GWY file from its magic header

        this is based on the file contents instead of the extension
        because a gwyddion file may still have its original .sxm filename
        """

        with open(self.file_path, "rb") as file:
            magic = file.read(4)

        return magic == b"GWYP"

    def _set_coordinates(self, x_m, y_m):
        """
        store lateral coordinates in the centered nanometer convention that
        is used in pyatoms
        """

        self.x_nm = np.asarray(x_m, dtype=float) * 1e9
        self.y_nm = np.asarray(y_m, dtype=float) * 1e9

        self.x_center_nm = (np.min(self.x_nm) + np.max(self.x_nm)) / 2
        self.y_center_nm = (np.min(self.y_nm) + np.max(self.y_nm)) / 2

        self.x_display_nm = self.x_nm - self.x_center_nm
        self.y_display_nm = self.y_nm - self.y_center_nm

        self.nx = len(self.x_nm)
        self.ny = len(self.y_nm)

        self.x_range_nm = float(np.max(self.x_display_nm) - np.min(self.x_display_nm))
        self.y_range_nm = float(np.max(self.y_display_nm) - np.min(self.y_display_nm))

    def _load_nanonis(self):
        """
        load a normal nanonis SXM file through xarray-nanonis
        """

        if not self.file_path.lower().endswith(".sxm"):
            raise ValueError(
                "The selected file is neither a Gwyddion GWY file "
                "nor a Nanonis .sxm file."
            )

        self.dataset = xr.open_dataset(self.file_path, engine="nanonis")

        self.channels = list(self.dataset.data_vars)

        self.directions = [str(value) for value in self.dataset["dir"].values]

        self._set_coordinates(self.dataset["x"].values, self.dataset["y"].values)

        self.metadata = dict(self.dataset.attrs)

    def _load_gwyddion(self):
        """
        load 2D gwyddion image fields

        gwyddion titles such as:
        
        Z (Forward)
        Z (Backward)
        Current (Forward)
        Current (Backward)

        are converted into the same channel/direction interfact that pyatoms
        already uses for Nanonis data
        """

        self._gwy_container = gwyfile.load(self.file_path)

        channels = []
        directions = []

        direction_pattern = re.compile(r"^(.*?)\s*\((forward|backward)\)\s*$", re.IGNORECASE)

        for field_index, title in gwyfile.util.find_datafields(self._gwy_container):
            title = str(title)

            match = direction_pattern.match(title)

            if match:
                channel = match.group(1).strip()
                direction = match.group(2).lower()

            else:
                # if the gwyddion title does not specify a scan direction then treat it as a forward image
                channel = title.strip()
                direction = "forward"

            key = (channel, direction)

            # if two data fields produce the same channel/direction pair then preserve the second one under its full title
            if key in self._gwy_fields:
                channel = title.strip()
                key = (channel, direction)

            field = self._gwy_container[f"/{field_index}/data"]

            self._gwy_fields[key] = field
            self._gwy_field_ids[key] = field_index

            if channel not in channels:
                channels.append(channel)

            if direction not in directions:
                directions.append(direction)

        if len(self._gwy_fields) == 0:
            raise ValueError("No 2D image channels were found in the Gwyddion file.")


        self.channels = channels

        # keep the familiar order in the gui
        self.directions = [direction for direction in ("forward", "backward") if direction in directions]

        self.directions.extend(direction for direction in directions if direction not in self.directions)

        # use the first image to initialize image dimensions
        first_field = next(iter(self._gwy_fields.values()))

        self._set_gwyddion_coordinates(first_field)

        self.metadata = {
            "format": "Gwyddion GWY",
            "source_file": self._gwy_container.get("/filename", self.file_name)
        }


    def _set_gwyddion_coordinates(self, field):
        """
        convert gwyddion lateral dimensions to meters and then pass them
        through the normal pyatoms coordinate handling
        """

        data = np.asarray(field.data)

        if data.ndim != 2:
            raise ValueError("The selected Gwyddion channel is not two-dimensional.")

        ny, nx = data.shape

        if field.xreal is None or field.yreal is None:
            raise ValueError("The Gwyddion channel does not contain physical scan dimensions.")

        unit = ""

        if field.si_unit_xy is not None:
            unit = str(field.si_unit_xy.unitstr)

        unit_scale = {
            "": 1.0,
            "m": 1.0,
            "nm": 1e-9,
            "um": 1e-6,
            "µm": 1e-6,
            "pm": 1e-12,
            "Å": 1e-10,
        }

        if unit not in unit_scale:
            raise ValueError(f"Unsupported Gwyddion lateral unit '{unit}'.")

        scale = unit_scale[unit]

        x_real_m = float(field.xreal) * scale
        y_real_m = float(field.yreal) * scale

        x_offset_m = float(field.xoff) * scale
        y_offset_m = float(field.yoff) * scale

        x_m = np.linspace(x_offset_m, x_offset_m + x_real_m, nx)
        y_m = np.linspace(y_offset_m, y_offset_m + y_real_m, ny)

        self._set_coordinates(x_m, y_m)


    def load_channel(self, channel="Z", direction="forward"):
        if channel not in self.channels:
            raise ValueError(f"Channel '{channel}' not found in dataset. Available channels: {self.channels}")

        if direction not in self.directions:
            raise ValueError(f"Direction '{direction}' not found in dataset. Available directions: {self.directions}")


        if self.file_format == "gwyddion":
            key = (channel, direction)

            if key not in self._gwy_fields:
                available_directions = [current_direction for current_channel, current_direction in self._gwy_fields if current_channel == channel]
                raise ValueError(
                    f"Direction '{direction}' is not available "
                    f"for channel '{channel}'. "
                    f"Available directions: {available_directions}"
                )

            field = self._gwy_fields[key]

            # gwyddion stores image rows from top to bottom
            # BUT pyatoms plots experimental images using origin = "lower" so we have to flip vertically
            data = np.flipud(np.asarray(field.data, dtype=float))

            self._set_gwyddion_coordinates(field)

            field_index = self._gwy_field_ids[key]

            self.metadata = {
                "format": "Gwyddion GWY",
                "source_file": self._gwy_container.get("/filename", self.file_name),
                "channel_title": self._gwy_container.get(f"/{field_index}/data/title", channel),
            }

            field_metadata = self._gwy_container.get(f"/{field_index}/meta")

            if field_metadata is not None:
                self.metadata.update(dict(field_metadata))

        else:
            data = self.dataset[channel].sel(dir=direction)

            data = np.asarray(data.values, dtype=float)

        self.channel = channel
        self.direction = direction

        # load raw data
        self.raw = np.asarray(data, dtype=float).copy()

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

    