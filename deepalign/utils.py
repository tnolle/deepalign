# Copyright 2020 Timo Nolle
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
# ==============================================================================

import numpy as np
import seaborn as sns

from deepalign import fs
from deepalign.enums import Axis
from deepalign.enums import Heuristic
from deepalign.enums import Strategy

microsoft_colors = sns.color_palette(['#01b8aa', '#374649', '#fd625e', '#f2c80f', '#5f6b6d',
                                      '#8ad4eb', '#fe9666', '#a66999', '#3599b8', '#dfbfbf'])


def reverse(x, m=None):
    if m is None:
        m = x != 0
    rx = np.copy(x)
    for i, j in enumerate(m):
        rx[i, j] = rx[i, j][::-1]
    return rx


def log_probs(y_true, y_pred, m=1):
    log_probs = np.log(gather(y_pred, np.atleast_3d(y_true))[:, :, 0]) * m
    cum_sum = np.cumsum(log_probs, axis=-1) * m
    return log_probs, cum_sum


def top_k_acc(x, y, k=1):
    match = np.any(np.all(x == y[:, None, :], axis=-1)[:, :k], axis=-1)
    correct = np.where(match)[0]
    incorrect = np.where(~match)[0]
    return correct.shape[0] / (correct.shape[0] + incorrect.shape[0])


def gather(x, query):
    base = lambda i, n, v: tuple([v if i == j else 1 for j in range(n)])
    idx = np.zeros_like(query, dtype=int)
    for i, dim in enumerate(x.shape[:-1]):
        idx += np.arange(x.shape[i]).reshape(base(i, x.ndim, x.shape[i])) * np.product(x.shape[i + 1:]).astype(int)
    idx += query
    return x.ravel()[idx].reshape(query.shape)


def align(x, s, constant_values=0):
    if x.ndim != 3:
        x = np.atleast_3d(x)
    if s > 0:
        x = np.pad(x[:, s:], ((0, 0), (0, s), (0, 0)), 'constant', constant_values=constant_values)
    else:
        x = np.pad(x[:, :s], ((0, 0), (-s, 0), (0, 0)), 'constant', constant_values=constant_values)
    return x


def to_targets(x):
    return align(x, 1)[:, :, 0] * (x != 0).astype(int)


def download_pretrained_models():
    from io import BytesIO
    from zipfile import ZipFile
    import requests
    from deepalign.fs import ROOT_DIR

    r = requests.get('https://github.com/tnolle/deepalign/releases/download/1.0.0/pretrained-models.zip',
                     allow_redirects=True)
    file = ZipFile(BytesIO(r.content))
    file.extractall(ROOT_DIR)

    return 'Download finished, check your `.out` folder.'
