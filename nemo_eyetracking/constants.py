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

from pathlib import Path

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / 'data'

N_JOBS = 4  # Number of cores/threads for multiprocessing

SAMPLING_RATE = 60        # Eye tracker sampling rate
INVALID_WINDOW_MS = 5000  # Minimum window (ms) with stable gaze in order to reject participant demographics
INVALID_WINDOW_FV = 1000   # Minimum window (ms) with stable gaze in order to reject participant freeviewing

SCREENSIZE = (1920, 1080)  # 16:9
DISPSIZE = (598, 336)      # mm
SCREENDIST = 800           # mm

HESSELS_SAVGOL_LEN  = 0           # Window length of Savitzky-Golay filter in pre-processing
HESSELS_THR         = 10e12       # Initial slow/fast phase threshold
HESSELS_LAMBDA      = 2.5         # Number of standard deviations (default 2.5)
HESSELS_MAX_ITER    = 50         # Max iterations for threshold adaptation (default 200)
HESSELS_WINDOW_SIZE = 8 * SAMPLING_RATE      # Threshold adaptation window (default 8 seconds) * sampling rate
HESSELS_MIN_AMP     = 1.0         # Minimal amplitude of fast candidates for merging slow candidates (default 1.0)
HESSELS_MIN_FIX     = .06         # Minimal fixation duration in seconds (default .06)

BINRANGES = [(6, 11), (12, 17), (18, 23), (24, 29), (30, 35), (36, 41), (42, 47), (48, 53), (54, 59)]
BINSTRINGS = ['06-11', '12-17', '18-23', '24-29', '30-35', '36-41', '42-47', '48-53', '54-59']
