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

from deepalign.anomalydetection.utils import label_collapse
from deepalign.anomalydetection.utils import max_collapse
from deepalign.anomalydetection.utils.heuristic import best_heuristic
from deepalign.anomalydetection.utils.heuristic import elbow_heuristic
from deepalign.anomalydetection.utils.heuristic import lowest_plateau_heuristic
from deepalign.anomalydetection.utils.heuristic import ratio_heuristic
from deepalign.enums import Base
from deepalign.enums import Class
from deepalign.enums import Heuristic
from deepalign.enums import Strategy


class Binarizer(object):
    def __init__(self, result, mask, features, targets=None):
        self.result = result
        self._mask = mask
        self.features = features
        self._targets = targets

        # Try to fix dimensions
        if self._mask.shape != self.result.scores.shape:
            if len(self._mask) != len(self.result.scores.shape):
                self._mask = np.expand_dims(self._mask, axis=-1)
            self._mask = np.repeat(self._mask, self.result.scores.shape[-1], axis=-1)

        self.targets = None
        if self._targets is not None:
            self.targets = dict((a, self.apply_mask(label_collapse(self._targets, axis=a))) for a in [0, 1, 2])

    def apply_mask(self, a):
        if len(a.shape) == 1:
            m = self._mask[:, 0, 0]
        elif len(a.shape) == 2:
            m = self._mask[:, :, 0]
        else:
            m = self._mask
        return np.ma.array(a, mask=m)

    def get_targets(self, axis=2):
        return self.targets.get(axis)

    def correct_shape(self, tau, strategy):
        tau = np.asarray(tau)
        if strategy == Strategy.POSITION:
            tau = tau[:, None]
        if strategy == Strategy.POSITION_ATTRIBUTE:
            tau = tau.reshape(*self.result.scores.shape[1:])
        return tau

    def split_by_strategy(self, a, strategy):
        if strategy == Strategy.SINGLE:
            return [a]
        elif isinstance(a, list):
            if strategy == Strategy.POSITION:
                return [[_a[:, i:i + 1] for _a in a] for i in range(len(a[0][0]))]
            elif strategy == Strategy.ATTRIBUTE:
                return [[_a] for _a in a]
            elif strategy == Strategy.POSITION_ATTRIBUTE:
                return [[_a[:, i:i + 1]] for i in range(len(a[0][0])) for _a in a]
        else:
            if strategy == Strategy.POSITION:
                return [a[:, i:i + 1, :] for i in range(a.shape[1])]
            elif strategy == Strategy.ATTRIBUTE:
                return [a[:, :, i:i + 1] for i in range(a.shape[2])]
            elif strategy == Strategy.POSITION_ATTRIBUTE:
                return [a[:, i:i + 1, j:j + 1] for i in range(a.shape[1]) for j in range(a.shape[2])]

    def get_grid_candidate_taus(self, a, steps=20, axis=0):
        """G in the paper."""
        return np.linspace(max_collapse(a, axis=axis).min() - .001, a.max(), steps)

    def get_candidate_taus(self, a, axis=0):
        a = max_collapse(a, axis=axis).compressed()
        if (len(a) == 0):
            return np.array([0, 0, 0, 0, 0])
        a_min = a.min()
        a_max = a.max()
        if a_max > a_min:
            a = (a_max - a) / (a_max - a_min)
        a = 2 * (a / 2).round(2)
        if a_max > a_min:
            a = a_max - a * (a_max - a_min)
        a = np.sort(np.unique(a))
        a[0] -= .001
        if len(a) < 5:
            a = np.linspace(a_min - .001, a_max, 5)
        return a

    def get_legacy_tau(self, scores, heuristic=Heuristic.DEFAULT, strategy=Strategy.SINGLE, axis=0):
        if heuristic == Heuristic.DEFAULT:
            return np.array([0.5])

        if not isinstance(scores, np.ma.MaskedArray):
            scores = self.apply_mask(scores)

        alpha = None
        if strategy == Strategy.SINGLE:
            alpha = np.array([scores.mean()])
        elif strategy == Strategy.ATTRIBUTE:
            alpha = scores.mean(axis=1).mean(axis=0).data
        elif strategy == Strategy.POSITION:
            alpha = scores.mean(axis=2).mean(axis=0).data[:, None]
        elif strategy == Strategy.POSITION_ATTRIBUTE:
            alpha = scores.mean(axis=0).data

        taus = self.get_grid_candidate_taus(scores / alpha, axis=axis)
        tau = None
        if heuristic == Heuristic.BEST:
            y_true = self.get_targets(axis=axis)
            tau = best_heuristic(taus=taus, theta=self.legacy_binarize, y_true=y_true, alpha=alpha, scores=scores,
                                 axis=axis)

        if heuristic == Heuristic.RATIO:
            tau = ratio_heuristic(taus=taus, theta=self.legacy_binarize, scores=scores, axis=axis, alpha=alpha)

        if heuristic in [Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP]:
            tau = elbow_heuristic(taus=taus, theta=self.legacy_binarize, scores=scores, axis=axis,
                                  alpha=alpha)[heuristic]

        if heuristic in [Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT]:
            tau = lowest_plateau_heuristic(taus=taus, theta=self.legacy_binarize, scores=scores, axis=axis,
                                           alpha=alpha)[heuristic]

        return tau * alpha

    def get_tau(self, scores, heuristic=Heuristic.DEFAULT, strategy=Strategy.SINGLE, axis=0, taus=None):
        if heuristic == Heuristic.DEFAULT:
            return np.array([0.5])

        if not isinstance(scores, np.ma.MaskedArray):
            scores = self.apply_mask(scores)

        scores = self.split_by_strategy(scores, strategy)

        if heuristic in [Heuristic.MEAN, Heuristic.MEDIAN]:
            scores = [max_collapse(s, axis=axis) for s in scores]
            if heuristic == Heuristic.MEAN:
                return self.correct_shape([np.mean(s[np.round(s, 1) > 0]) for s in scores], strategy)
            elif heuristic == Heuristic.MEDIAN:
                return self.correct_shape([np.median(s[np.round(s, 1) > 0]) for s in scores], strategy)

        if taus is None:
            taus = [self.get_candidate_taus(s, axis=axis) for s in scores]
        else:
            taus = [taus] * len(scores)

        tau = None
        if heuristic == Heuristic.BEST:
            y_trues = self.split_by_strategy(self.get_targets(axis=2), strategy)
            y_trues = [label_collapse(y, axis=axis) for y in y_trues]
            tau = [best_heuristic(taus=t, theta=self.threshold_binarize, y_true=y, scores=s, axis=axis)
                   for s, t, y in zip(scores, taus, y_trues)]

        if heuristic == Heuristic.RATIO:
            tau = [ratio_heuristic(taus=t, scores=s, theta=self.threshold_binarize, axis=axis)
                   for s, t in zip(scores, taus)]

        if heuristic in [Heuristic.ELBOW_DOWN, Heuristic.ELBOW_UP]:
            tau = [elbow_heuristic(taus=t, scores=s, theta=self.threshold_binarize, axis=axis)[heuristic]
                   for s, t in zip(scores, taus)]

        if heuristic in [Heuristic.LP_LEFT, Heuristic.LP_MEAN, Heuristic.LP_RIGHT]:
            tau = [lowest_plateau_heuristic(taus=t, scores=s, theta=self.threshold_binarize, axis=axis)[heuristic]
                   for s, t in zip(scores, taus)]

        return self.correct_shape(tau, strategy)

    def legacy_binarize(self, scores, tau, alpha, axis=0):
        # Apply the threshold function (Theta in the paper) using alpha as a scaling factor
        return self.threshold_binarize(tau=tau * alpha, scores=scores, axis=axis)

    def threshold_binarize(self, tau, scores, axis=0):
        # Apply the threshold function (Theta in the paper)
        predictions = np.array(scores.data > tau, dtype=int)

        # Apply mask
        predictions = np.ma.array(predictions, mask=scores.mask)

        # Positive axis flatten predictions
        if axis in [0, 1]:
            predictions = label_collapse(predictions, axis=axis)

        return predictions

    def binarize(self, scores=None, tau=None, base=None, heuristic=None, strategy=None, go_backwards=False,
                 return_parameters=False, axis=2, heuristic_axis=None):

        if heuristic_axis is None:
            heuristic_axis = axis

        if scores is None:
            if go_backwards:
                scores = self.result.scores_backward
            else:
                scores = self.result.scores

        if not isinstance(scores, np.ma.MaskedArray):
            scores = self.apply_mask(scores)

        # Get baseline threshold (tau in the paper)
        if tau is None or heuristic != Heuristic.MANUAL:
            if base == Base.LEGACY:
                tau = self.get_legacy_tau(scores=scores, heuristic=heuristic, strategy=strategy, axis=heuristic_axis)
            else:
                tau = self.get_tau(scores=scores, heuristic=heuristic, strategy=strategy, axis=heuristic_axis)

        # Apply the threshold function (Theta in the paper)
        predictions = self.threshold_binarize(scores=scores, tau=tau, axis=axis)

        if return_parameters:
            return predictions, tau

        return predictions

    @staticmethod
    def get_scores(probabilities):
        scores = np.zeros_like(probabilities)
        for i in range(scores.shape[2]):
            p = probabilities[:, :, i:i + 1]
            _p = np.copy(probabilities)
            _p[_p <= p] = 0
            scores[:, :, i] = _p.sum(axis=2)
        return scores
