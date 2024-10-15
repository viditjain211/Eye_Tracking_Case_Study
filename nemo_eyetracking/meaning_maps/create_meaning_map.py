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

import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image, ImageDraw, ImageOps

from constants import DATA_DIR, ROOT_DIR, SCREENSIZE

meaning_dir = ROOT_DIR / 'data' / 'meaning_maps'

# Load necessary files
df = pd.read_csv(meaning_dir / '_study_results' / 'processed_data.csv')
coords = pd.read_csv(meaning_dir / '_coordinates' / 'coords.csv')
im = Image.open(DATA_DIR / 'image_hd.png')

# Create 1080x1920x2 matrix. We'll store a counter in the last dimension
meaning_map = np.zeros((SCREENSIZE[1], SCREENSIZE[0], 2), dtype=float)

# Retrieve each rating
for i in range(len(df)):
    im_name = df.iloc[i]['image']
    response = float(df.iloc[i]['Scaled Response'])

    # Find the coordinates for which this rating was made
    coords_row = coords.loc[coords['im'] == im_name]
    x1 = coords_row['x1'].values[0]
    y1 = coords_row['y1'].values[0]
    x2 = coords_row['x2'].values[0]
    y2 = coords_row['y2'].values[0]

    if x1 < 0: x1 = 0
    if y1 < 0: y1 = 0
    if x2 > SCREENSIZE[0]: x2 = SCREENSIZE[0]
    if y2 > SCREENSIZE[1]: y2 = SCREENSIZE[1]

    # Add the response value and increment a counter
    meaning_map[y1:y2, x1:x2, 0] += response
    meaning_map[y1:y2, x1:x2, 1] += 1

# Normalize the responses by dividing by the number of occurences
agg_map = meaning_map[:, :, 0] / meaning_map[:, :, 1]

with open(meaning_dir / '_study_results' / '2dmap.p', 'wb') as f:
    pickle.dump(meaning_map, f)

with open(meaning_dir / '_study_results' / 'aggmap.p', 'wb') as f:
    pickle.dump(agg_map, f)

# Plot
plt.figure(figsize=(7.5, 4.2))

plt.imshow(im)
sns.heatmap(agg_map, alpha=0.9, cbar=False, cmap='magma')

plt.xlim((0, 1920))
plt.ylim((1080, 0))

plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig(meaning_dir / '_overlaid' / 'meaning_map_discrete.png', dpi=600)
plt.show()


# Plot blurred heatmap
from scipy.ndimage import gaussian_filter

gauss_map = gaussian_filter(agg_map, sigma=45)  # 1 dva is approximately 45 pixels

plt.figure(figsize=(7.5, 4.2))

plt.imshow(im)
sns.heatmap(gauss_map, alpha=0.9, cbar=False, cmap='magma')

plt.xlim((0, 1920))
plt.ylim((1080, 0))

plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.savefig(meaning_dir / '_overlaid' / 'meaning_map_gauss.png', dpi=600)
plt.show()
