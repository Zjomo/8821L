'''
Code for the PyTorch implementation of
"DeepFocus: a Few-Shot Microscope Slide Auto-Focus using a Sample-invariant CNN-based Sharpness Function"

Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
Written by Adrian Shajkofci <adrian.shajkofci@idiap.ch>,
All rights reserved.

This file is part of DeepFocus.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
3. Neither the name of mosquitto nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from toolbox import pickle_load
import glob
import os

# Disable LaTeX to avoid dependency issues
plt.rc('text', usetex=False)
font = {'family': 'serif',
        'weight': 'normal',
        'size': 16}
fig = plt.figure(figsize=(5, 3), dpi=300)
ax = fig.gca()
matplotlib.rc('font', **font)
legends = []

for file in glob.glob('errors_simulations_*.pkl'):

    all_errors = pickle_load(file)
    all_errors = np.asarray(all_errors)
    range_of_acquisitions = np.arange(3, 8, 1).tolist()

    filename = file[19:-4]
    legends.append(filename.replace('_',''))
    filtered_all_errors = []
    filtered_all_errors_mean = []
    filtered_all_errors_std =[]
    for i in range(all_errors.shape[0]):
        err = all_errors[i]
        upper_quartile = np.percentile(err, 75)
        lower_quartile = np.percentile(err, 25)
        IQR = (upper_quartile - lower_quartile) * 0.5
        quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

        err = err[np.where((err>= quartileSet[0]) & (err<= quartileSet[1]))]

        mean = err.mean(axis=0)
        var = err.std(axis=0)
        filtered_all_errors.append(err)
        filtered_all_errors_mean.append(mean)
        filtered_all_errors_std.append(var)

    print('Method = {}, mean = {} +/- {}'.format(filename, filtered_all_errors_mean, filtered_all_errors_std))
    print('Method = {}, mean mean = {} +/- {}'.format(filename, np.asarray(filtered_all_errors_mean).mean(), np.asarray(filtered_all_errors_std).mean()))

    plt.errorbar(range_of_acquisitions, filtered_all_errors_mean, yerr=filtered_all_errors_std, fmt='-x',
                 elinewidth=2, capsize=2)
plt.legend(legends, prop={'size': 10})

plt.xlabel('Autofocus iterations', fontsize=12)
plt.ylabel('Error (um)', fontsize=12)
plt.tight_layout()
plt.ylim([-0.5, 35])
ax.set_xticks(np.arange(3,11,1))
plt.rc('grid', linestyle="dotted", color='black')
plt.gcf().subplots_adjust(bottom=0.21, right=0.95)

plt.grid(True)
# Save to current directory instead of hardcoded path
output_path = os.path.join(os.getcwd(), 'simulations_plot.png')
plt.savefig(output_path, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

plt.show()