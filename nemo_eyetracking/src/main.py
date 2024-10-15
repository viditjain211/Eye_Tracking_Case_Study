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
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from constants import DATA_DIR, N_JOBS, ROOT_DIR
from src.plots import plot_rms, plot_timeseries
from utils.dataloader_helpers import (check_for_valid_gaze_freeviewing,
                                      check_for_valid_gaze_overall,
                                      enumerate_and_concat_fixations,
                                      format_single_df, get_filepaths_csv,
                                      get_filepaths_txt, get_participant_info,
                                      load_file_as_df, pp_info_add_valid,
                                      run_hessels_classifier)


# TEST
def check_isi():
    # Check inter-sample interval
    dfs, files = load_data(pre_process=False)

    isi = []
    for df in dfs:
        ts = np.array(df['time_from_start'])
        diffs = np.diff(ts)
        diffs = diffs[diffs < 0.03]  # Remove large gaps, because those are not ISIs but just missing data
        isi.append(np.nanmean(diffs))

    print(isi)

    isi = np.array(isi)
    print(np.mean(isi), 'mean')
    print(np.min(isi), 'min')
    print(np.max(isi), 'max')


def process_plots(dfs: List[pd.DataFrame], files: List[Path]) -> None:
    print('Making event detection plots')
    pp_info = pd.read_csv(ROOT_DIR / 'results' / 'participant_info.csv')
    pp_info['ID'] = pp_info['ID'].astype(str)

    # DEBUG
    # dfs = dfs[400:410]
    # files = files[400:410]

    # Change folder and filetype in the filename specification
    if '/raw_data' in str(files[0]):
        files = [str(f).replace('/raw_data', '/pre_processed') for f in files]
        files = [str(f).replace('.txt', '.csv') for f in files]
    elif '/fixation_events' in str(files[0]):
        files = [str(f).replace('/fixation_events', '/pre_processed') for f in files]

    # Make timeseries plots. Only show in IDE if it's < 30 files
    for df, file in zip(dfs, files):
        pinfo_id = pp_info.loc[pp_info['ID'] == Path(file).stem]

        if list(pinfo_id['Valid Freeviewing'])[0]:
            plot_timeseries(df, file, show=True if len(dfs) < 30 else False)


def multiproc_dataloader(files: List[Path]) -> List[pd.DataFrame]:
    print('Loading data')
    dfs = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(delayed(load_file_as_df)(f) for f in files)
    return dfs


def multiproc_format_df(dfs: List[pd.DataFrame], files: List[Path], save_to_csv: bool) -> List[pd.DataFrame]:
    print('Formatting dataframes')

    saveto = [save_to_csv] * len(dfs)
    dfs = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(delayed(format_single_df)(df, f, s) for df, f, s in zip(dfs, files, saveto))

    return dfs


def multiproc_valid_gaze(dfs: List[pd.DataFrame]) -> List[bool]:
    print('Checking for valid gaze')
    valids = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(delayed(check_for_valid_gaze_overall)(df) for df in dfs)
    return valids


def multiproc_valid_gaze_fv(dfs: List[pd.DataFrame]) -> List[bool]:
    print('Checking for valid gaze in freeviewing')
    valids = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(delayed(check_for_valid_gaze_freeviewing)(df) for df in dfs)
    return valids


def multiproc_hessels_classifier(dfs: List[pd.DataFrame], files: List[Path]) -> List[pd.DataFrame]:
    print('Running Hessels classifier')
    fixations = Parallel(n_jobs=N_JOBS, backend='loky', verbose=True)(delayed(run_hessels_classifier)(df, f) for df, f in zip(dfs, files))

    return fixations


def load_data(pre_process=True, n: int = None) -> Tuple[List[pd.DataFrame], List[Path]]:
    if pre_process:
        files = get_filepaths_txt(DATA_DIR / 'raw_data')

        if n is not None:
            files = files[0:n]

        # Load data and retrieve participant info
        dfs = multiproc_dataloader(files)
        pp_info = get_participant_info(dfs, files)

        # Format files and save immediately
        dfs = multiproc_format_df(dfs, files, save_to_csv=True)

        # Check whether each dataset is valid and add it to the pp_info dataframe
        valid_fv_list = multiproc_valid_gaze_fv(dfs)
        valid_list = multiproc_valid_gaze(dfs)
        pp_info_add_valid(pp_info, valid_list, valid_fv_list)

    else:
        # Load already pre-processed data
        files = get_filepaths_csv(DATA_DIR / 'pre_processed')

        if n is not None:
            files = files[0:n]

        dfs = multiproc_dataloader(files)

    return dfs, files


def load_fixation_events(dfs: List[pd.DataFrame] = None, files: List[Path] = None) -> Tuple[List[pd.DataFrame],
                                                                                            List[Path]]:
    if dfs is not None:

        # Change folder and filetype in the filename specification
        if '/raw_data' in str(files[0]):
            files = [str(f).replace('/raw_data', '/pre_processed') for f in files]
            files = [str(f).replace('.txt', '.csv') for f in files]

        events = multiproc_hessels_classifier(dfs, files)

    else:
        files = get_filepaths_csv(DATA_DIR / 'fixation_events')
        events = multiproc_dataloader(files)

    return events, files


def main(n: int = None) -> None:
    """
    :param n: number of files to process. default = None; extract all. Specifying n will pick the first N files
    :return:
    """

    # Load and process. Set pre_process=False if preprocessing has already been done (saves time)
    # dfs, files = load_data(pre_process=True, n=n)
    dfs, files = load_data(pre_process=False, n=n)

    # Load or process fixation events. Supply no dataframes if you want to load already saved events
    # events, files = load_fixation_events(dfs, files)
    events, files = load_fixation_events()

    # Make timeseries plots of fixation detection
    # process_plots(dfs, files)

    # Filter out saccades, and enumerate all fixation events per participant (i.e., 1st, 2nd landings, etc.).
    # Then combine all participants into one big df.
    _ = enumerate_and_concat_fixations(events, files)

    # Make other plots
    # plot_rms(dfs, files, compute=True)


if __name__ == '__main__':
    main()
