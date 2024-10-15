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

import time
from datetime import datetime
from pathlib import Path
from random import sample
from typing import Any, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import median_absolute_deviation

import constants
from constants import (DATA_DIR, HESSELS_SAVGOL_LEN, INVALID_WINDOW_FV,
                       INVALID_WINDOW_MS, ROOT_DIR, SAMPLING_RATE)
from utils.helperfunctions import gender_str_convert
from utils.hessels_classifier import classify_hessels2020


def get_filepaths_txt(data_path: Path) -> List[Path]:
    files = sorted(list(data_path.glob('*.txt')))
    print(f'{len(files)} .txt files found')

    return files


def get_filepaths_csv(data_path: Path) -> List[Path]:
    files = sorted(list(data_path.glob('*.csv')))
    print(f'{len(files)} .csv files found')

    return files


def load_files_as_df(files: List[Path]) -> List[pd.DataFrame]:
    # Multi-dataframe container function for load_file_as_df
    dfs = []

    for file in files:
        try:
            dfs.append(load_file_as_df(file))

        except Exception as e:
            print(f'Error loading {str(file)}: {e}')

    return dfs


def load_file_as_df(file: Path) -> pd.DataFrame:
    # Load file as .txt or as .csv
    try:
        if file.suffix == '.txt':
            with open(file, 'r') as f:
                text = f.read()

            text = text.split('\n')
            df = pd.DataFrame(text)

        elif file.suffix == '.csv':
            df = pd.read_csv(file)

        else:
            df = None
            print(f'Cannot load {file.suffix} filetype')

        return df

    except Exception as e:
        print(f'Error loading {str(file)}: {e}')


def save_dataframes_as_csv(dfs: List[pd.DataFrame], filenames: List[Path], save_path: Path, suffix: str = '') -> None:
    for df, f in zip(dfs, filenames):
        df.to_csv(Path(save_path / f'{Path(f).stem}{suffix}.csv'))


def _split_column(x) -> List[Any]:
    x = str(x)

    x = x.replace('(', '')
    x = x.replace(')', '')
    row = x.split(',')

    # Some rows are short or empty
    if len(row) < 3:
        return [np.nan] * 5

    # If row contains event, move it to the end and keep x/y columns open
    # (i.e., [e, nan, nan, nan, nan timestamp, message])
    if row[0] == 'e':
        split_message = row[1].split(':')
        row = [row[0], np.nan, np.nan, np.nan, np.nan, row[2], split_message[0], split_message[1]]
    else:
        # Check whether GazePoint is valid, otherwise assign nan's
        if row[5] is True or row[5] == 'True':
            xgaze = float(row[6])
            ygaze = constants.SCREENSIZE[1] - float(row[7])  # Flip the y-origin for the raw signal
            pupil = float(row[9])

            if ygaze < 0 or ygaze > 1080 or xgaze < 0 or xgaze > 1920:
                xgaze, ygaze = np.nan, np.nan

            else:
                # Scale x by 1.2308, but from the center outwards - meaning 1.1154 in either direction.
                # We do this because there was a scaling issue in the raw gaze coordinates
                xgaze -= (constants.SCREENSIZE[0]) / 2
                xgaze *= 1.1154
                xgaze += (constants.SCREENSIZE[0]) / 2

                # Scale y, but only upward
                ygaze *= 1.2308

            row = [row[0], float(row[1]), float(row[2]), xgaze, ygaze, row[3], '', '', True, pupil]

        else:
            row = [row[0], float(row[1]), float(row[2]), np.nan, np.nan, row[3], '', '', False, np.nan]

    return row


def format_single_df(df: pd.DataFrame, filename, save_to_csv) -> pd.DataFrame:
    df = df.iloc[5:]

    # Now separate by commas
    new_columns = [_split_column(x) for x in list(df.iloc[:, 0])]
    new_df = pd.DataFrame(new_columns, columns=['label',
                                                'x_old', 'y_old',
                                                'x', 'y',
                                                'timestamp', 'event', 'message',
                                                'valid_gaze',
                                                'pupil_size'])

    # Add more convenient timestamps
    new_df['datetime'] = new_df['timestamp'].apply(convert_timestamp, args=(False,))
    new_df['unix_time'] = new_df['timestamp'].apply(convert_timestamp)

    start_time = list(new_df['unix_time'])[0]
    new_df['time_from_start'] = new_df['unix_time'].apply(lambda x: x - start_time)

    # Drop the filtered gaze coordinates
    new_df = new_df.drop(['x_old', 'y_old'], axis=1)

    if save_to_csv:
        assert filename is not None, 'format_single_df(): Cannot save to csv if no filenames are supplied'
        save_path = ROOT_DIR / 'data' / 'pre_processed'
        save_dataframes_as_csv([new_df], [filename], save_path)

    return new_df


def get_participant_info(dfs: List[pd.DataFrame], files: [List[Path]]) -> pd.DataFrame:
    info = {'ID': [], 'Gender': [], 'DoB': [],
            'ROI1': [], 'ROI2': [], 'ROI3': []}

    for df, ID in zip(dfs, files):
        rows = list(np.arange(5))

        info['ID'     ].append(str(Path(ID).stem))
        info['Gender' ].append(gender_str_convert(df.iloc[rows[0], 0]))
        info['DoB'    ].append(df.iloc[rows[1], 0])
        info['ROI1'   ].append(df.iloc[rows[2], 0])
        info['ROI2'   ].append(df.iloc[rows[3], 0])
        info['ROI3'   ].append(df.iloc[rows[4], 0])

    info = pd.DataFrame(info)

    info.to_csv(ROOT_DIR / 'results' / 'participant_info.csv')

    return info


def pp_info_add_valid(info: pd.DataFrame, valid, valid_fv) -> None:
    info['Valid'] = valid
    info['Valid Freeviewing'] = valid_fv

    info.to_csv(ROOT_DIR / 'results' / 'participant_info.csv')


def convert_timestamp(timestamp: str, as_unix: bool = True) -> Union[datetime, float]:
    # Timestamps should be strings, e.g., '20220129125024936' -> '2022/01/29 12:50:24 936ms'
    timestamp = str(timestamp)

    if timestamp == 'nan' or timestamp == 'None':
        return np.nan

    year, month, day = timestamp[0:4], timestamp[4:6], timestamp[6:8]
    hour, minute, second, ms = timestamp[8:10], timestamp[10:12], timestamp[12:14], timestamp[14:17]

    # datetime timestamp
    date_time = datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), int(ms) * 1000)

    # Unix timestamp
    unix_time = date_time.timestamp()

    if as_unix:
        return unix_time
    else:
        return date_time


def check_for_valid_gaze_overall(df: pd.DataFrame):
    # Check whether there is a period of invalid gaze the size of INVALID_WINDOW_MS.
    # This means someone likely left the eyetracker and we can't trust the demographics
    sample_window = int(INVALID_WINDOW_MS / SAMPLING_RATE)

    valid = list(df['valid_gaze'])

    messages = np.array(df['message'])

    i = 0
    while i < len(valid) - sample_window:
        window_start = i
        window_end = i + sample_window

        # Mark the end of the experiment
        if 'ScreenVideoFeedback' in messages[window_start:window_end]:
            break

        # There is a window of size sample_window which only contains False; so someone likely left the tracker
        if True not in valid[window_start:window_end]:
            return False

        i += 1

    return True


def check_for_valid_gaze_freeviewing(df: pd.DataFrame):
    # Check whether there is a period of invalid gaze the size of INVALID_WINDOW_FV.
    # This means someone likely left the eyetracker, so we can't trust the freeviewing data
    sample_window = int(INVALID_WINDOW_FV / SAMPLING_RATE)

    start = 0

    # Check which row marks the end of freeviewing
    events = np.array(df['event'])
    messages = np.array(df['message'])
    end = np.argwhere((events == 'ScreenStart') & (messages == 'ScreenInstructionPostGame')).ravel()[0]

    # If there's more missing data than the sample window allows, mark as False.
    # 600 is the default number of rows (10 seconds, 60Hz)
    if end - start < (600 - sample_window):
        return False

    valid = list(df['valid_gaze'])[start:end]

    i = 0
    while i < len(valid) - sample_window:
        window_start = i
        window_end = i + sample_window

        # There is a window of size sample_window which only contains False; so someone likely left the tracker
        if True not in valid[window_start:window_end]:
            return False

        i += 1

    return True


def determine_valid(filename: Path) -> bool:
    # Determine from pp_info whether a participant is considered valid
    pp_info = pd.read_csv(ROOT_DIR / 'results' / 'participant_info.csv')
    pp_info['ID'] = pp_info['ID'].astype(str)

    ID = str(Path(filename).stem)
    is_valid_df = pp_info.loc[pp_info['ID'] == ID]

    if len(is_valid_df) > 0:
        is_valid = is_valid_df['Valid Freeviewing'].values[0]

        return is_valid

    else:
        return False


def filter_df(df: pd.DataFrame) -> pd.DataFrame:
    # Select appropriate indices
    # (row 0 until ScreenInstructionPostGame appears in message column. This way we get the first 10 seconds)
    if 'e' in list(df['label'].unique()):
        start = 0
        end = np.argwhere(np.array(df['message']) == 'ScreenInstructionPostGame').ravel()[0]
        df = df.iloc[start : end - 1]

    df = apply_savgol(df)

    return df


def apply_savgol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies a savitsky-golay filter to the raw gaze signal columns
    :param df:
    :return:
    """

    if HESSELS_SAVGOL_LEN > 0:
        x = np.array(df['x_exact'])
        y = np.array(df['y_exact'])

        try:
            x = savgol_filter(x, HESSELS_SAVGOL_LEN, 2, mode='nearest')
            y = savgol_filter(y, HESSELS_SAVGOL_LEN, 2, mode='nearest')

            df.loc[:, 'x_exact'] = x
            df.loc[:, 'y_exact'] = y

        except:
            pass

    return df


def run_hessels_classifier(df: pd.DataFrame, filename: Path) -> pd.DataFrame:
    # Determine from pp_info whether the pp is valid
    is_valid = determine_valid(filename)

    if is_valid:
        df = filter_df(df)

        if 'Unnamed: 0' in list(df.columns):
            df = df.drop(['Unnamed: 0'], axis=1)

        df = df.drop(['label', 'timestamp',
                      'event', 'message', 'valid_gaze',
                      'pupil_size',
                      'datetime', 'unix_time'], axis=1)

        df.columns = ['x', 'y', 'time']

        fixations = classify_hessels2020(df)

    else:
        # If not valid, make empty df
        fixations = pd.DataFrame(columns=['onset', 'offset', 'duration', 'avg_x', 'avg_y', 'label'])

    save_dataframes_as_csv([fixations], [filename], save_path=DATA_DIR / 'fixation_events')

    return fixations


def enumerate_and_concat_fixations(events: List[pd.DataFrame], files) -> pd.DataFrame:
    """
    Load all fixation_events files and combine them into one big df
    :param events:
    :param files:
    :return:
    """

    if '/pre_processed' in str(files[0]):
        files = [str(f).replace('/pre_processed', '/fixation_events') for f in files]

    enumerated_dfs = []
    total_fixes = []
    fix_dur = []
    for df, file in zip(events, files):
        ID = Path(file).stem

        fix_only = df.loc[df['label'] == 'FIXA'].reset_index()
        fix_only['ID'] = [ID] * len(fix_only)
        fix_only['Order'] = np.arange(len(fix_only))

        enumerated_dfs.append(fix_only)

        if len(fix_only) > 0:
            total_fixes.append(len(fix_only) - 1)  # Don't count the 0th fixation, which is on the fix cross
            fix_dur.append(np.nanmedian(fix_only.iloc[0:]['duration']))

    result = pd.concat(enumerated_dfs)
    result.to_csv(DATA_DIR / 'compiled_fixations.csv')

    print(f'Median {np.nanmedian(total_fixes).round(2)} fixations per person '
          f'({round(np.nanmedian(total_fixes) / 10, 2)} per second). '
          f'MAD = {median_absolute_deviation(total_fixes)}, '
          f'{np.nanmedian(total_fixes) - median_absolute_deviation(total_fixes)}. '
          f'Median duration = {np.nanmedian(fix_dur).round(4)} seconds (N = {len(total_fixes)})')

    plt.figure()
    plt.hist(total_fixes)  # , bins=max(total_fixes))
    plt.xlabel('Number of fixations')
    plt.ylabel(f'Count (N = {len(total_fixes)})')
    plt.title(f'Median = {np.nanmedian(total_fixes).round(2)}')
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'fix_distribution.png')
    plt.show()

    return result
