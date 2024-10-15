"""
nemo_eyetracking
Copyright (C) 2022 Utrecht University

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

from math import atan2, dist
from typing import Any, List, Tuple, Union

import numpy as np

from constants import BINRANGES, BINSTRINGS, DISPSIZE, SCREENDIST, SCREENSIZE


def gender_str_convert(x: str) -> str:
    if 'FEMALE' in x:
        return 'FEMALE'
    elif 'MALE' in x:
        return 'MALE'
    else:
        return 'OTHER'


def px_to_dva(px: Union[float, np.ndarray, List[Any]], orientation='x') -> Union[float, np.ndarray]:
    """
    Converts pixel distance to degrees of visual angle
    :param px: single float or array of pixel values
    :param orientation: 'x' or 'y'
    :return: Outputs in the same format as px param
    """

    if orientation == 'x':
        pixel_size_mm = DISPSIZE[0] / SCREENSIZE[0]
    elif orientation == 'y':
        pixel_size_mm = DISPSIZE[1] / SCREENSIZE[1]
    else:
        raise ValueError('px_to_dva: supply x or y orientation')

    angle = np.radians(1.0 / 2.0)  # Angle of 1
    mm_on_screen = np.tan(angle) * SCREENDIST  # mm size of 1 degree visual angle
    pix_per_deg = (mm_on_screen / pixel_size_mm) * 2

    return px / pix_per_deg


def get_displacement(x: Union[np.ndarray, List[Any]], y: Union[np.ndarray, List[Any]]) -> List[float]:
    # Computes the euclidean distance from each datapoint to the next
    disp = []
    for p1x, p1y, p2x, p2y in zip(x[:-1], y[:-1], x[1:], y[1:]):
        disp.append(dist((p1x, p1y), (p2x, p2y)))

    return disp


def compute_rms(x: Union[np.ndarray, List[Any]], y: Union[np.ndarray, List[Any]]) -> np.ndarray:
    disp = get_displacement(x, y)           # Pixel displacement between samples
    dva = px_to_dva(np.array(disp)) ** 2    # Squared displacement between samples
    rms = np.sqrt(np.nanmean(dva))          # Root mean of squared displacement

    return rms


def dob_to_age(ID: Union[int, str], DoB: Union[int, str]) -> int:
    # Because we don't know pp's birth month (only year), we
    # assume that everyone's birthday is halfway through the year: July 1st
    year = int(str(ID[0:4]))
    month = int(str(ID[4:6]))

    if month >= 7:
        age = 1 + year - int(DoB)
    else:
        age = year - int(DoB)

    return age


def age_to_bin(age: Union[int, float]) -> str:
    for binrange, binstr in zip(BINRANGES, BINSTRINGS):
        if binrange[0] <= age <= binrange[1]:
            return binstr

    return 'None'
