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
from random import choices, shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

from constants import DATA_DIR, SCREENSIZE
from utils.helperfunctions import compute_meaning_grid_pos

im_path = DATA_DIR / 'image_hd.png'
im = Image.open(im_path)

save_dir = DATA_DIR / 'meaning_maps'

# Degrees of visual angle in pixels
one_deg = 44.83
oneandhalf_deg = one_deg * 1.5
three_deg = one_deg * 3
seven_deg = one_deg * 7

coords = {'im': [], 'x1': [], 'y1': [], 'x2': [], 'y2': []}

im_names = []

for deg in [one_deg, oneandhalf_deg, three_deg, seven_deg]:
    im_copy = im.copy()
    draw = ImageDraw.Draw(im_copy)

    deg = np.round(deg).astype(int)
    locs = compute_meaning_grid_pos(SCREENSIZE, deg)

    # Draw image with overlaid circles
    for (x, y) in locs:
        x1, x2 = x - deg, x + deg
        y1, y2 = y - deg, y + deg

        draw.ellipse([x1, y1, x2, y2])
        draw.ellipse([x - 5, y - 5, x + 5, y + 5], fill='red')

    im_copy.save(save_dir / '_overlaid' / f'_{deg}_overlaid.png')

    # Crop images
    for i, (x, y) in enumerate(locs):
        x1, x2 = x - deg, x + deg
        y1, y2 = y - deg, y + deg

        mask = Image.new('L', im.size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse([x1, y1, x2, y2], fill=255)

        output = ImageOps.fit(im, mask.size, centering=(0.5, 0.5))
        output.putalpha(mask)

        output = output.crop([x1, y1, x2, y2])
        output = output.resize((256, 256))

        im_name = f'{deg}_cropped_{i}.png'
        output.save(save_dir / im_name)

        coords['im'].append(im_name)
        coords['x1'].append(x1)
        coords['y1'].append(y1)
        coords['x2'].append(x2)
        coords['y2'].append(y2)

        im_names.append(im_name)

coords = pd.DataFrame(coords)
coords.to_csv(save_dir / '_coordinates' / f'coords.csv')

print(f'{len(im_names)} total stimuli')  # 649 (1145 with 1/3/7 deg)

np.random.seed(42)
all_im_occurences = {n: 0 for n in im_names}  # Just a tracker

shuffle(im_names)

# Create excels with k stimuli each
k = 200

prev_start = -k
prev_end = 0

# Keep a rolling counter.
# Start at 0:k, then increment with k and rollover when start or end is greater than list length
# (e.g., grab 900:1145 and 0:54 to still get 300 values).
for i in range(4):
    start, end = prev_start + k, prev_end + k

    if start >= len(im_names) - 1:
        start -= len(im_names)
        end = start + k
        im_samples = im_names[start:end]

    elif end >= len(im_names) - 1:
        im_samples = im_names[start:len(im_names)]

        end -= len(im_names)
        im_samples += im_names[0: end]

    else:
        im_samples = im_names[start:end]

    prev_start, prev_end = start, end

    # Update the tracker
    for n in im_samples:
        all_im_occurences[n] += 1

    # Create the excel file. Start with two rows (consent and instruction) and end with debrief.
    # In between, fill in the shuffled k stimulus names.
    d = {'randomise_blocks': [''],
         'randomise_trials': [''],
         'display': ['instructions'],
         'image': ['']}

    for j in range(k):
        d['randomise_blocks'].append('')
        d['randomise_trials'].append('1')
        d['display'].append('rating-display')
        d['image'].append(im_samples[j])

    d['randomise_blocks'].append('')
    d['randomise_trials'].append('')
    d['display'].append('debrief')
    d['image'].append('')

    df = pd.DataFrame(d)
    df.to_csv(save_dir / '_spreadsheets' / f'spreadsheet_{i + 1}.csv')


# Print the number of occurences just to double check
occurrences = sorted(all_im_occurences.items(), key=lambda x: x[1])
print('\nNumber of occurences, sorted from low to high:')
print(occurrences)
