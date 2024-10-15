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

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

from constants import ROOT_DIR

result_dir = ROOT_DIR / 'data' / 'meaning_maps' / '_study_results'
files = sorted(list(result_dir.glob('*.xlsx')))

# Read all result files
dfs = [pd.read_excel(f, engine='openpyxl') for f in files]
df = pd.concat(dfs)

# Filter for only rows with response values in it
df = df.loc[df['Zone Type'] == 'response_slider_endValue']

# Keep a limited set of columns
drop_cols = [c for c in list(df.columns) if c not in ['Participant Public ID',
                                                      'Spreadsheet',
                                                      'Response',
                                                      'image']]
df = df.drop(drop_cols, axis=1)

# Add a condition column
df['Condition'] = df['image'].apply(lambda x: str(x).split('_')[0])

# Scale the responses, per ID and per 'condition' (number of degrees).
# We do the latter (separate scaling) because large areas of an image will
# always be relatively meaningful as compared to small areas
df_list = []

IDs = list(df['Participant Public ID'].unique())
for ID in IDs:
    dfid = df.loc[df['Participant Public ID'] == ID]

    conditions = list(dfid['Condition'].unique())
    for condition in conditions:
        dfc = dfid.loc[dfid['Condition'] == condition].reset_index()

        # Get response values, scale, and add as new column
        responses = np.array(dfc['Response'])
        dfc['Scaled Response'] = scale(responses)

        df_list.append(dfc)

# Merge dataframes and save
df = pd.concat(df_list, ignore_index=True)
df.to_csv(result_dir / 'processed_data.csv')

# Make plots of response distributions to check for invalid participants
IDs = list(df['Participant Public ID'].unique())

for ID in IDs:
    responses = df.loc[df['Participant Public ID'] == ID]['Response'].values
    plt.figure()
    plt.hist(responses)
    plt.xlim((0, 100))
    plt.title(f'{ID}, {len(responses)} responses')
    plt.savefig(result_dir / 'response_plots' / f'{ID}.png', dpi=200)
    # plt.show()
    plt.close()


# Demographics
demo_dir = ROOT_DIR / 'data' / 'meaning_maps' / '_demographics'
demo_files = list(demo_dir.glob('*data_exp_*'))
demo_file = demo_files[0]
demo_df = pd.read_excel(demo_file, engine='openpyxl')

IDs = list(demo_df["Participant Public ID"].unique())
print(f'{len(IDs)} IDs: {IDs}')

demo_dict = {'ID': [], 'Gender': [], 'Age': []}

for ID in IDs:
    if ID in list(df['Participant Public ID'].unique()):
        demo = demo_df.loc[demo_df['Participant Public ID'] == ID]

        try:
            gender = demo.loc[demo['Question Key'] == 'Gender']['Response'].values[0]
            age = demo.loc[demo['Question Key'] == 'Age']['Response'].values[0]

            demo_dict['ID'].append(str(ID))
            demo_dict['Gender'].append(gender)
            demo_dict['Age'].append(int(age))

        except Exception as e:
            print(e, '\n', demo)
            demo_dict['ID'].append(ID)
            demo_dict['Gender'].append(np.nan)
            demo_dict['Age'].append(np.nan)

    else:
        print(f'ID {ID} not found in task data!')

demo_dict = pd.DataFrame(demo_dict)
demo_dict.to_excel(demo_dir / 'demographics.xlsx')

print(f'Demographics \n'
      f'Gender      : {Counter(demo_dict["Gender"])} \n' 
      f'Age (median): {np.median(demo_dict["Age"])} \n'
      f'Age (SD)    : {np.std(demo_dict["Age"])}')


