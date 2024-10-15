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
from random import shuffle
from typing import List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from constants import BINSTRINGS, ROOT_DIR, SAMPLING_RATE
from utils.helperfunctions import (age_to_bin, compute_rms, dob_to_age,
                                   gender_str_convert, get_displacement,
                                   px_to_dva)


def plot_timeseries(df: pd.DataFrame, filename, show=True):
    """
    Plot a timeseries with periods of fixation overlaid, if possible.
    Comment/uncomment to plot either x/y or displacement
    :param df: pre-processed data
    :param filename:
    :param show: whether to show plots as output (True), or save only (False)
    :return:
    """

    x = np.array(df['x']).ravel()
    y = np.array(df['y']).ravel()
    t = np.array(df['time_from_start']).ravel()

    try:
        fixations = pd.read_csv(str(filename).replace('/pre_processed', '/fixation_events'))
        fixations = fixations.loc[fixations['label'] == 'FIXA']
        starts = np.array(fixations['onset'])
        ends = np.array(fixations['offset'])

        if len(starts) == 0:
            starts, ends = [0], [0]

    except Exception as e:
        print(e)
        starts, ends = [0], [0]

    if len(x) > 0:
        try:
            labels = ['x', 'y', 'Fixation']
            palette = list(sns.color_palette('deep', 10))

            f = plt.figure(figsize=(10, 5))

            # Make subplots
            ax = []
            ax.append(f.add_subplot(1, 1, 1))

            # X/Y
            h0 = ax[0].plot(t, x, label=labels[0],
                            linewidth=1,
                            color=palette[0],
                            marker='o', markersize=1.5
                            )
            h1 = ax[0].plot(t, y, label=labels[2],
                            color=palette[2],
                            linestyle='--',
                            linewidth=1,
                            marker='o', markersize=1.5
                            )

            ax[0].set_ylim((0, 2000))
            ax[0].set_yticks([0, 540, 1080, 1920])
            ax[0].set_ylabel('Gaze position (x/y)')
            ax[0].set_xlabel('Time (s)')
            ax[0].set_xlim((0, 10))

            # Displacement
            # displacement = get_displacement(xe, ye)
            # displacement = px_to_dva(displacement) * SAMPLING_RATE
            # h2 = ax[1].plot(t[1:], displacement, label=labels[4],
            #                 color='gray',
            #                 alpha=.5,
            #                 # linewidth=1,
            #                 linewidth=0,
            #                 zorder=-20
            #                 )
            #
            # ax[1].set_ylim((0, 300))  # np.max(displacement) * 2
            # ax[1].set_yticks([0, 50, 100, 150, 200, 250])
            # ax[1].set_ylabel('Velocity (°/s)')
            #
            # ax[1].yaxis.set_label_position("right")
            # ax[1].yaxis.tick_right()
            # ax[1].set_yticks([])
            #
            # ax[1].set_xticks([])
            # ax[1].set_xlim((0, 10))
            # ax[1].set_xlabel('')

            # Boxes which indicate periods of fixation
            for i, (s, e) in enumerate(zip(starts, ends)):
                h3 = ax[0].axvspan(s, e, color=palette[4], alpha=.05, label=labels[2], zorder=-30)

            ax[0].legend([h0[0], h1[0], h3], labels, loc='upper right')

            plt.title(Path(filename).stem)
            plt.tight_layout()

            plt.savefig(
                str(filename).replace('/data/pre_processed', '/results/plots/timeseries').replace('.csv', '.png'),
                dpi=200)

            if show:
                plt.show()

            plt.close()

        except ValueError:
            pass


def plot_rms(dfs: List[pd.DataFrame] = None, files: List[Path] = None, compute: bool = False):
    """
    Computes and plots (and saves to csv) the RMS over all participants
    :param dfs: Optional list of dataframes. If not supplied, function will try to load pre-computed RMS values
    :param files:
    :param compute:
    :return:
    """
    if compute or dfs is not None:
        print('Computing RMS... This may take a while')
        pp_info = pd.read_csv(ROOT_DIR / 'results' / 'participant_info.csv')
        pp_info['ID'] = pp_info['ID'].astype(str)

        # Locate participants with 'valid' demographics
        pp_info_valid = pp_info.loc[pp_info['Valid'] == True]
        valid_ids = list(pp_info_valid['ID'].unique())

        d = {'ID': [], 'RMS': [], 'Loss': [], 'Valid': [], 'Age': [], 'Bin': [], 'Gender': []}

        for df, file in zip(dfs, files):
            ID = Path(file).stem

            # Convert date of birth to age, get gender
            dfid = pp_info.loc[pp_info['ID'] == ID]
            age = dfid['DoB'].values[0]
            age = dob_to_age(ID, age)
            gender = dfid['Gender'].values[0]
            gender = gender_str_convert(gender).title()

            # Take only the first 10s (freeviewing)
            df = df.iloc[0:600]
            x = df['x']
            y = df['y']
            vg = np.array(df['valid_gaze'])

            # Use a window to compute RMS per window, then take the median
            samples = 12  # window of 200 ms
            stepsize = 2  # 32 ms
            x = list(x)
            y = list(y)

            rms_values = []
            for st in np.arange(0, len(x) - samples, step=stepsize):
                x_ = x[st: st + samples]
                y_ = y[st: st + samples]

                # Compute RMS for this window
                rms = compute_rms(x_, y_)
                rms_values.append(rms)

            rms = np.nanmedian(rms_values)

            loss_ = np.nansum(vg) / len(df)
            loss = 1 - loss_

            # Add to a dict
            d['ID'].append(ID)

            if ID in valid_ids:
                d['Valid'].append(True)
                d['RMS'].append(rms.round(3))
                d['Loss'].append(loss * 100)
                d['Age'].append(age)
                d['Bin'].append(age_to_bin(age))
                d['Gender'].append(gender)
            else:
                d['Valid'].append(False)
                d['RMS'].append(rms.round(3))
                d['Loss'].append(loss * 100)
                d['Age'].append(np.nan)
                d['Bin'].append('None')
                d['Gender'].append(np.nan)

        d = pd.DataFrame(d)
        d = d.sort_values(by='Bin')

        d.to_csv(ROOT_DIR / 'results' / 'rms.csv')

    else:
        d = pd.read_csv(ROOT_DIR / 'results' / 'rms.csv')

    d = d.loc[d['Bin'] != 'None']
    d = d.loc[d['Gender'] != 'Other']
    d = d.sort_values(by='Bin')

    print(f'Median overall RMS = {np.nanmedian(d["RMS"]).round(3)} (SD = {np.nanstd(d["RMS"]).round(3)})')
    print(f'Mean overall Loss = {np.nanmean(d["Loss"]).round(3)} (SD = {np.nanstd(d["Loss"]).round(3)})')

    # Binned ages
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=d, x='Bin', y='RMS', sort=False,
                 estimator=np.nanmedian)
    plt.ylabel('RMS (°)')
    plt.xlabel('Age (bin)')

    handles = BINSTRINGS.copy()
    handles[0] = '6-11'
    plt.xticks(BINSTRINGS, handles)

    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'rms_age.pdf', dpi=600)
    plt.show()
    plt.close()

    # Binned ages but loss
    plt.figure(figsize=(6, 4))
    sns.lineplot(data=d, x='Bin', y='Loss', sort=False,
                 )
    plt.ylabel('Loss (%)')
    plt.xlabel('Age (bin)')

    handles = BINSTRINGS.copy()
    handles[0] = '6-11'
    plt.xticks(BINSTRINGS, handles)

    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'loss_age.pdf', dpi=600)
    plt.show()
    plt.close()

    d_age = d.groupby(['Bin']).agg({'Loss': np.nanmean,
                                    'RMS': np.nanmedian}).reset_index()
    d_age.to_csv(ROOT_DIR / 'results' / f'loss_rms_age.csv')

    # Gender
    plt.figure(figsize=(6, 4))
    sns.pointplot(data=d, x='Gender', y='RMS', sort=False,
                  capsize=0.15,
                  estimator=np.nanmedian)
    plt.ylabel('RMS (°)')
    plt.xlabel('Gender')

    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'rms_gender.pdf', dpi=600)
    plt.show()
    plt.close()

    # Gender but loss
    plt.figure(figsize=(6, 4))
    sns.pointplot(data=d, x='Gender', y='Loss', sort=False,
                  capsize=0.15, )
    plt.ylabel('Loss (%)')
    plt.xlabel('Gender')

    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'loss_gender.pdf', dpi=600)
    plt.show()
    plt.close()

    d_gen = d.groupby(['Gender']).agg({'Loss': np.nanmean,
                                       'RMS': np.nanmedian}).reset_index()
    d_gen.to_csv(ROOT_DIR / 'results' / f'loss_rms_gender.csv')

    # Scatter
    dg = d.groupby(['Bin']).agg({'Loss': np.nanmean,
                                 'RMS': np.nanmedian}).reset_index()
    plt.figure(figsize=(6, 4))
    sns.scatterplot(data=dg, x='RMS', y='Loss', hue='Bin')

    plt.xlabel('RMS (°)')
    plt.ylabel('Loss (%)')

    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'loss_rms_scatter.pdf', dpi=600)
    plt.show()


def plot_age_nss_colourcoded():
    # Load NSS for gender and age. These need some (manual) reconfiguring coming from the saleval repo
    df_age = pd.read_csv(ROOT_DIR / 'results' / 'age_nss.csv')
    df_age['Age'] = df_age['Age'].apply(lambda x: x.replace('age', ''))

    f = plt.figure(figsize=(7.5, 5))
    ax_age1 = f.add_subplot(1, 1, 1)

    ###
    # AGE
    ###
    # Draw unique dotted line for each saliency model
    pal = list(sns.color_palette('deep', n_colors=10))
    ls_list = ['-'] * 10
    pal += list(sns.color_palette('muted', n_colors=10))
    ls_list += ['-.'] * 10
    pal += list(sns.color_palette('tab10', n_colors=10))
    ls_list += ['--'] * 10

    j = 0
    for m in sorted(list(df_age['Model'].unique())):
        if m == 'Meaning map':
            ls = ':'
            color = 'black'
        elif m == 'Single observer':
            ls = '-.'
            color = 'black'
        elif m == 'Central bias' or m == 'Fixation map':
            ls = '--'
            color = 'black'
        else:
            ls = '-'
            color = pal[j]
            j += 1

        lw = 1.2

        dfm = df_age.loc[df_age['Model'] == m]
        ax_age1.plot(dfm['Age'],
                     dfm['NSS deviation'],
                     color=color,
                     linestyle=ls, linewidth=lw,
                     zorder=0,
                     label=m
                     )

    # Add the 0 line
    ax_age1.axhline(0, color='gray', linestyle='--', zorder=-50)

    # Do formatting
    ax_age1.set_ylabel('Delta prediction (NSS)', fontsize=9)

    ax_age1.set_ylim((-0.12, 0.15))
    ax_age1.set_yticks([-0.1, -0.05, 0, 0.05, 0.1],
                       ['-0.1', '-0.05', '0', '0.05', '0.1'],
                       fontsize=8)

    ax_age1.set_xlabel('Age (years)', fontsize=9)
    # Change the age bin from '06-11' to '6-11'. Do this now, because using '6-11' doesn't sort the x-axis properly
    ax_age1.set_xticks(np.arange(9),
                       ['6-11', '12-17', '18-23', '24-29', '30-35', '36-41', '42-47', '48-53', '54-59'],
                       fontsize=8)

    ax_age1.spines['top'].set_visible(False)
    ax_age1.spines['right'].set_visible(False)

    ax_age1.legend(loc='upper left', frameon=False, title=None, fontsize=8, ncol=5)

    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'Fig2_colourcoded.png', dpi=600)
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'Fig2_colourcoded.pdf', dpi=600)
    plt.show()


def plot_age_nss_proprotion():
    # Load NSS for gender and age. These need some (manual) reconfiguring coming from the saleval repo
    df_age = pd.read_csv(ROOT_DIR / 'saleval' / 'results' / 'tables' / 'nss_by_age.csv')

    for col in BINSTRINGS:
        max_nss = df_age.loc[df_age['Model'] == 'Fixation map'][col].values[0]
        min_nss = df_age.loc[df_age['Model'] == 'Single observer'][col].values[0]
        df_age[col] = ((np.array(df_age[col]) - min_nss) / (max_nss - min_nss)) * 100

    df_age = df_age.loc[df_age['Model'] != 'Fixation map']
    df_age = df_age.loc[df_age['Model'] != 'Single observer']

    df_age.to_csv(ROOT_DIR / 'saleval' / 'results' / 'tables' / 'nss_by_age_prop.csv')

    df_age = pd.melt(df_age, id_vars=['Model'], value_vars=BINSTRINGS, var_name='Age', value_name='NSS')

    f = plt.figure(figsize=(7.5, 5))
    ax_age1 = f.add_subplot(1, 1, 1)

    ###
    # AGE
    ###
    # Draw unique dotted line for each saliency model
    pal = list(sns.color_palette('deep', n_colors=10))
    ls_list = ['-'] * 10
    pal += list(sns.color_palette('muted', n_colors=10))
    ls_list += ['-.'] * 10
    pal += list(sns.color_palette('tab10', n_colors=10))
    ls_list += ['--'] * 10

    j = 0
    for m in sorted(list(df_age['Model'].unique())):
        if m == 'Meaning map':
            ls = ':'
            color = 'black'
        elif m == 'Single observer':
            ls = '-.'
            color = 'black'
        elif m == 'Central bias' or m == 'Fixation map':
            ls = '--'
            color = 'black'
        else:
            ls = ls_list[j]
            color = pal[j]
            j += 1

        lw = 1.2

        dfm = df_age.loc[df_age['Model'] == m]
        ax_age1.plot(dfm['Age'],
                     dfm['NSS'],
                     color=color,
                     linestyle=ls, linewidth=lw,
                     zorder=0,
                     label=m
                     )

    sns.pointplot(data=df_age, x='Age', y='NSS',
                  estimator=np.nanmean,
                  markers='d',
                  scale=1.5,
                  join=False,
                  color='black',
                  capsize=0.15,
                  errwidth=1,
                  ax=ax_age1)

    # Do formatting
    ax_age1.set_ylabel('NSS (% of maximum achieveable)', fontsize=10)

    ax_age1.set_ylim((-55, 90))
    ax_age1.set_yticks(np.arange(-50, 70, step=10),
                       fontsize=9)

    ax_age1.set_xlabel('Age (years)', fontsize=10)
    # Change the age bin from '06-11' to '6-11'. Do this now, because using '6-11' doesn't sort the x-axis properly
    ax_age1.set_xticks(np.arange(9),
                       ['6-11', '12-17', '18-23', '24-29', '30-35', '36-41', '42-47', '48-53', '54-59'],
                       fontsize=9)

    ax_age1.spines['top'].set_visible(False)
    ax_age1.spines['right'].set_visible(False)

    ax_age1.legend(loc='upper left', frameon=False, title=None, fontsize=8, ncol=5)

    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'Fig2_proportion.png', dpi=600)
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'Fig2_proportion.pdf', dpi=600)
    plt.show()


def plot_gender_age_nss():
    # Load NSS for gender and age. These need some (manual) reconfiguring coming from the saleval repo
    df_gen = pd.read_csv(ROOT_DIR / 'results' / 'gender_nss.csv')
    df_age = pd.read_csv(ROOT_DIR / 'results' / 'age_nss.csv')
    df_age['Age'] = df_age['Age'].apply(lambda x: x.replace('age', ''))

    df_gen = df_gen.loc[df_gen['Model'] != 'Fixation map']
    df_gen = df_gen.loc[df_gen['Model'] != 'Central bias']
    df_gen = df_gen.loc[df_gen['Model'] != 'Meaning map']
    df_gen = df_gen.loc[df_gen['Model'] != 'Single observer']
    df_age = df_age.loc[df_age['Model'] != 'Fixation map']
    df_age = df_age.loc[df_age['Model'] != 'Central bias']
    df_age = df_age.loc[df_age['Model'] != 'Meaning map']
    df_age = df_age.loc[df_age['Model'] != 'Single observer']

    f = plt.figure(figsize=(7.5, 3))

    gs = gridspec.GridSpec(1, 3)
    ax_gen = f.add_subplot(gs[0])
    ax_age1 = f.add_subplot(gs[1:3])

    ###
    # GENDER
    ###
    sns.violinplot(data=df_gen, x='Gender', y='NSS deviation', hue='Gender', ax=ax_gen,
                   palette=[sns.color_palette('Set2')[0], sns.color_palette('Set2')[1]],
                   saturation=.5,
                   inner=None,
                   linewidth=.5,
                   zorder=0,
                   dodge=False)
    sns.stripplot(data=df_gen, x='Gender', y='NSS deviation', ax=ax_gen,
                  color='gray',
                  marker='.',
                  jitter=0,
                  zorder=1,
                  size=5)

    # MEDIANS
    sns.pointplot(data=df_gen, x='Gender', y='NSS deviation',
                  estimator=np.nanmean,
                  markers='d',
                  join=False,
                  color='black',
                  capsize=0.15,
                  errwidth=0.8,
                  zorder=10e20,
                  ax=ax_gen)

    # Add dashed line at 0
    l1 = ax_gen.axhline(0, color='gray', linestyle='--', zorder=-10)

    # Do formatting stuff
    ax_gen.set_ylabel('Delta prediction (NSS)', fontsize=9)

    ax_gen.set_ylim((-0.12, 0.1))
    ax_gen.set_yticks([-0.1, -0.05, 0, 0.05, 0.1],
                      ['-0.1', '-0.05', '0', '0.05', '0.1'],
                      fontsize=8)

    ax_gen.set_xlabel('Gender', fontsize=9)
    ax_gen.set_xticks([0, 1], ['Male', 'Female'], fontsize=8)

    ax_gen.get_legend().remove()
    ax_gen.spines['top'].set_visible(False)
    ax_gen.spines['right'].set_visible(False)

    ###
    # AGE
    ###
    # Draw unique dotted line for each saliency model
    for m in list(df_age['Model'].unique()):
        dfm = df_age.loc[df_age['Model'] == m]
        l3 = ax_age1.plot(dfm['Age'],
                          dfm['NSS deviation'],
                          color='darkgray',  # alpha=.5,
                          linestyle=':', linewidth=.7,
                          zorder=0
                          )

    # MEDIANS
    sns.pointplot(data=df_age, x='Age', y='NSS deviation',
                  estimator=np.nanmean,
                  markers='d',
                  join=False,
                  color='black',
                  capsize=0.3,
                  errwidth=0.8,
                  zorder=10e20,
                  ax=ax_age1)

    # Add the 0 line
    ax_age1.axhline(0, color='gray', linestyle='--', zorder=-50)

    # Do formatting
    ax_age1.set_ylim(ax_gen.get_ylim())
    ax_age1.set_yticks([])
    ax_age1.set_ylabel('')

    ax_age1.set_xlabel('Age (years)', fontsize=9)
    # Change the age bin from '06-11' to '6-11'. Do this now, because using '6-11' doesn't sort the x-axis properly
    ax_age1.set_xticks(np.arange(9),
                       ['6-11', '12-17', '18-23', '24-29', '30-35', '36-41', '42-47', '48-53', '54-59'],
                       fontsize=8)

    ax_age1.spines['top'].set_visible(False)
    ax_age1.spines['left'].set_visible(False)
    ax_age1.spines['right'].set_visible(False)

    # Legend formatting
    diamond = mlines.Line2D([], [], color='black', marker='d', linewidth=0)  # Custom marker
    ax_age1.legend([l3[0], diamond, l1],
                   ['Individual models', 'Mean (±95% CI)', 'Average (across bins)'],
                   loc='lower left',
                   frameon=False,
                   fontsize=9)

    plt.tight_layout()
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'Fig2.png', dpi=600)
    plt.savefig(ROOT_DIR / 'results' / 'plots' / 'Fig2.pdf', dpi=600)
    plt.show()


def plot_nss_per_fix_offset():
    dir = ROOT_DIR / 'num_fix'
    files = sorted(list(dir.rglob('nss_results.csv')))

    files = [f for f in files if '-8' in str(f)]

    dl = ['SAM', 'DeepGazeI', 'DeepGazeII', 'DeepGazeIIE', 'SALICON', 'SalGAN']

    nss = []
    for f in files:
        nfix = int(f.parent.name.replace('-8', ''))

        df = pd.read_csv(f, usecols=['Model', 'NSS', 'Improvement'])
        df['Number of fixations'] = [nfix] * len(df)
        df['DL'] = df['Model'].apply(lambda x: True if x in dl else False)
        df['Improvement'] = df['Improvement'].apply(lambda x: float(str(x).replace(' %', '')))

        nss.append(df)

    nss = pd.concat(nss, ignore_index=True)
    nss.to_csv(ROOT_DIR / 'num_fix' / 'nss_concat_offset.csv')

    nss_ = []
    for model in list(nss['Model'].unique()):
        nssmodel = nss.loc[nss['Model'] == model]
        endscore = nssmodel.loc[nssmodel['Number of fixations'] == 18]['Improvement'].values[0]
        nssmodel['endscore'] = [endscore] * len(nssmodel)
        nss_.append(nssmodel)

    nss = pd.concat(nss_, ignore_index=True)
    nss = nss.sort_values(by=['Number of fixations', 'endscore'], ascending=False).reset_index()

    pal = list(sns.color_palette('deep', n_colors=10))
    pal += list(sns.color_palette('muted', n_colors=10))
    pal += list(sns.color_palette('tab10', n_colors=10))

    ## Absolute scores
    f = plt.figure(figsize=(7.5, 5.5))
    ax = f.add_subplot(1, 1, 1)

    j = 0
    for i, model in enumerate(list(nss['Model'].unique())):
        nssmodel = nss.loc[nss['Model'] == model]

        if model == 'Meaning map':
            ls = ':'
            color = 'black'
        elif model == 'Single observer':
            ls = '-.'
            color = 'black'
        elif model == 'Central bias' or model == 'Fixation map':
            ls = '--'
            color = 'black'
        else:
            ls = '-'
            color = pal[j]
            j += 1

        ax.plot(nssmodel['Number of fixations'], nssmodel['Improvement'],
                label=model,
                color=color,
                ls=ls)

    ax.set_xticks(np.arange(9, 19, step=1), np.arange(9, 19, step=1))
    ax.set_xlim((8.5, 18.5))
    ax.set_xlabel('Fixation', fontsize=12)
    ax.set_ylabel('NSS to centre prior (%)', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    f.tight_layout()
    plt.savefig(ROOT_DIR / 'num_fix' / 'relative_per_fix_offset.pdf', dpi=600)
    plt.show()


def plot_nss_per_fix():
    dir = ROOT_DIR / 'num_fix'
    files = sorted(list(dir.rglob('nss_results.csv')))

    files = [f for f in files if not '-8' in str(f)]

    dl = ['SAM', 'DeepGazeI', 'DeepGazeII', 'DeepGazeIIE', 'SALICON', 'SalGAN']

    nss = []
    for f in files:
        nfix = int(f.parent.name)

        df = pd.read_csv(f, usecols=['Model', 'NSS', 'Improvement'])
        df['Number of fixations'] = [nfix] * len(df)
        df['DL'] = df['Model'].apply(lambda x: True if x in dl else False)
        df['Improvement'] = df['Improvement'].apply(lambda x: float(str(x).replace(' %', '')))

        nss.append(df)

    nss = pd.concat(nss, ignore_index=True)
    nss.to_csv(ROOT_DIR / 'num_fix' / 'nss_concat.csv')

    nss_ = []
    for model in list(nss['Model'].unique()):
        nssmodel = nss.loc[nss['Model'] == model]
        endscore = nssmodel.loc[nssmodel['Number of fixations'] == 18]['Improvement'].values[0]
        nssmodel['endscore'] = [endscore] * len(nssmodel)
        nss_.append(nssmodel)

    nss = pd.concat(nss_, ignore_index=True)
    nss = nss.sort_values(by=['Number of fixations', 'endscore'], ascending=False).reset_index()

    nss = nss.loc[nss['Model'] != 'Spatial distribution map']

    pal = list(sns.color_palette('deep', n_colors=10))
    pal += list(sns.color_palette('muted', n_colors=10))
    pal += list(sns.color_palette('tab10', n_colors=10))
    # shuffle(pal)

    ## Absolute scores
    f = plt.figure(figsize=(7.5, 5.5))
    ax = f.add_subplot(1, 1, 1)

    j = 0
    for i, model in enumerate(list(nss['Model'].unique())):
        nssmodel = nss.loc[nss['Model'] == model]

        if model == 'Meaning map':
            ls = ':'
            color = 'black'
        elif model == 'Single observer':
            ls = '-.'
            color = 'black'
        elif model == 'Central bias' or model == 'Fixation map':
            ls = '--'
            color = 'black'
        else:
            ls = '-'
            color = pal[j]
            j += 1

        ax.plot(nssmodel['Number of fixations'], nssmodel['NSS'],
                label=model,
                color=color,
                ls=ls)

    ax.set_xticks(np.arange(1, 19, step=1))
    ax.set_xlim((0, 18.5))
    ax.set_xlabel('Fixation', fontsize=12)
    ax.set_ylabel('NSS', fontsize=12)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    f.tight_layout()
    plt.savefig(ROOT_DIR / 'num_fix' / 'nss_per_fix.pdf', dpi=600)
    plt.show()

    ## Relative scores
    f = plt.figure(figsize=(7.5, 5.5))
    ax = f.add_subplot(1, 1, 1)

    j = 0
    for i, model in enumerate(list(nss['Model'].unique())):
        nssmodel = nss.loc[nss['Model'] == model]

        if model == 'Meaning map':
            ls = ':'
            color = 'black'
        elif model == 'Single observer':
            ls = '-.'
            color = 'black'
        elif model == 'Central bias' or model == 'Fixation map':
            ls = '--'
            color = 'black'
        else:
            ls = '-'
            color = pal[j]
            j += 1

        ax.plot(nssmodel['Number of fixations'], nssmodel['Improvement'],
                label=model,
                color=color,
                ls=ls)

    ax.set_xticks(np.arange(1, 19, step=1))
    ax.set_xlim((0.5, 24))
    ax.set_xlabel('Fixation', fontsize=12)
    ax.set_ylabel('NSS to centre prior (%)', fontsize=12)

    ax.legend(loc='upper right', frameon=False, title='Model', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    f.tight_layout()
    plt.savefig(ROOT_DIR / 'num_fix' / 'relative_per_fix.pdf', dpi=600)
    plt.show()


if __name__ == '__main__':
    plot_gender_age_nss()
    plot_age_nss_colourcoded()
    plot_age_nss_proprotion()
    plot_nss_per_fix()
    plot_nss_per_fix_offset()
    plot_rms()
