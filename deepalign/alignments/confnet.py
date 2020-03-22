#  Copyright 2020 Timo Nolle
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#  ==============================================================================

from time import time

import numpy as np
import tensorflow as tf

from deepalign import Dataset
from deepalign.alignments.bibs import bibs_step
from deepalign.alignments.bibs import build_alignments
from deepalign.alignments.bibs import build_beams
from deepalign.alignments.bibs import get_alignments
from deepalign.anomalydetection import AnomalyDetectionResult
from deepalign.anomalydetection import Binarizer
from deepalign.enums import AttributeType
from deepalign.enums import FeatureType
from deepalign.enums import Heuristic
from deepalign.enums import Strategy
from deepalign.utils import align
from deepalign.utils import gather
from deepalign.utils import log_probs
from deepalign.utils import reverse
from deepalign.utils import to_targets


def binet_scores_fn(features, predictions):
    sums = [1 - np.cumsum(np.sort(p, -1), -1) for p in predictions]
    indices = [(np.argsort(p, -1) == features[:, :, i:i + 1]).argmax(-1) for i, p in enumerate(predictions)]
    scores = np.zeros(features.shape)
    for (i, j, k), f in np.ndenumerate(features):
        if f != 0 and k < len(predictions):
            scores[i, j, k] = sums[k][i, j][indices[k][i, j]]
    return scores


class BINet(tf.keras.Model):
    abbreviation = 'binet'
    name = 'BINet'

    def __init__(self,
                 dataset,
                 latent_dim=None,
                 use_case_attributes=None,
                 use_event_attributes=None,
                 use_present_activity=None,
                 use_present_attributes=None,
                 use_attention=None):
        super(BINet, self).__init__()

        # Validate parameters
        if latent_dim is None:
            latent_dim = min(int(dataset.max_len * 10), 256)
        if use_event_attributes and dataset.num_attributes == 1:
            use_event_attributes = False
            use_case_attributes = False
        if use_present_activity and dataset.num_attributes == 1:
            use_present_activity = False
        if use_present_attributes and dataset.num_attributes == 1:
            use_present_attributes = False

        # Parameters
        self.latent_dim = latent_dim
        self.use_case_attributes = use_case_attributes
        self.use_event_attributes = use_event_attributes
        self.use_present_activity = use_present_activity
        self.use_present_attributes = use_present_attributes
        self.use_attention = use_attention

        # Single layers
        self.fc = None
        if self.use_case_attributes:
            self.fc = tf.keras.Sequential([
                tf.keras.layers.Dense(latent_dim // 8),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(latent_dim, activation='linear')
            ])

        self.rnn = tf.keras.layers.GRU(latent_dim, return_sequences=True, return_state=True)

        # Layer lists
        self.fc_inputs = []
        self.rnn_inputs = []
        self.outs = []

        inputs = zip(dataset.attribute_dims, dataset.attribute_keys, dataset.attribute_types, dataset.feature_types)
        for dim, key, t, feature_type in inputs:
            if t == AttributeType.CATEGORICAL:
                voc_size = int(dim + 1)  # we start at 1, 0 is padding
                emb_dim = np.clip(voc_size // 10, 2, 10)
                embed = tf.keras.layers.Embedding(input_dim=voc_size, output_dim=emb_dim, mask_zero=True)
            else:
                embed = tf.keras.layers.Dense(1, activation='linear')

            if feature_type == FeatureType.CASE:
                self.fc_inputs.append(embed)
            else:
                self.rnn_inputs.append(embed)
                out = tf.keras.layers.Dense(dim + 1, activation='softmax')
                self.outs.append(out)

    def call(self, inputs, training=False, return_state=False, initial_state=None):
        if not isinstance(inputs, list):
            inputs = [inputs]

        split = len(self.rnn_inputs)

        rnn_x = inputs[:split]
        fc_x = inputs[split:]

        fc_embeddings = []
        for x, input_layer in zip(fc_x, self.fc_inputs):
            if isinstance(input_layer, tf.keras.layers.Dense):
                x = x[:, None]
            x = input_layer(x)
            fc_embeddings.append(x)

        if len(fc_embeddings) > 0:
            if len(fc_embeddings) > 1:
                fc_embeddings = tf.concat(fc_embeddings, axis=-1)
            else:
                fc_embeddings = fc_embeddings[0]

        fc_output = None
        if not isinstance(fc_embeddings, list):
            fc_output = self.fc(fc_embeddings)

        rnn_embeddings = []
        for x, input_layer in zip(rnn_x, self.rnn_inputs):
            x = input_layer(x)
            rnn_embeddings.append(x)

        if len(rnn_embeddings) > 0:
            if len(rnn_embeddings) > 1:
                rnn_embeddings = tf.concat(rnn_embeddings, axis=-1)
            else:
                rnn_embeddings = rnn_embeddings[0]

        if initial_state is not None:
            rnn, h = self.rnn(rnn_embeddings, initial_state=initial_state)
        elif fc_output is not None:
            if len(fc_output.shape) == 3:
                fc_output = fc_output[:, 0]
            rnn, h = self.rnn(rnn_embeddings, initial_state=fc_output)
        else:
            rnn, h = self.rnn(rnn_embeddings)

        outputs = []
        for i, out in enumerate(self.outs):
            x = rnn
            if i > 0:
                if self.use_present_attributes:
                    x = tf.concat([x, *[tf.pad(e[:, 1:x.shape[1]], [(0, 0), (0, 1), (0, 0)], 'constant', 0)
                                        for j, e in enumerate(rnn_embeddings) if i != j]], axis=-1)
                elif self.use_present_activity:
                    x = tf.concat([x, tf.pad(rnn_embeddings[0][:, 1:x.shape[1]], [(0, 0), (0, 1), (0, 0)], 'constant', 0)],
                                  axis=-1)
            x = out(x)
            outputs.append(x)

        if return_state:
            return outputs, h

        return outputs

    def score(self, features, predictions):
        for i, prediction in enumerate(predictions):
            p = np.pad(prediction[:, :-1], ((0, 0), (1, 0), (0, 0)), mode='constant')
            p[:, 0, features[i][0, 0]] = 1
            predictions[i] = p
        return binet_scores_fn(np.dstack(features), predictions)

    def detect(self, dataset):
        if isinstance(dataset, Dataset):
            features = dataset.features
        else:
            features = dataset
        predictions = self.predict(features)
        if not isinstance(predictions, list):
            predictions = [predictions]
        return AnomalyDetectionResult(scores=self.score(features, predictions), predictions=predictions)


class ConfNet:
    abbreviation = 'confnet'
    name = 'ConfNet'

    def __init__(self, dataset, latent_dim=None, use_case_attributes=None, use_event_attributes=None):
        super(ConfNet, self).__init__()

        self.use_case_attributes = use_case_attributes
        self.use_event_attributes = use_event_attributes

        self.net_f = BINet(dataset=dataset,
                           latent_dim=latent_dim,
                           use_case_attributes=use_case_attributes,
                           use_event_attributes=use_event_attributes)
        self.net_b = BINet(dataset=dataset,
                           latent_dim=latent_dim,
                           use_case_attributes=use_case_attributes,
                           use_event_attributes=use_event_attributes)

        self.net_f.compile(tf.keras.optimizers.Adam(), 'sparse_categorical_crossentropy')
        self.net_b.compile(tf.keras.optimizers.Adam(), 'sparse_categorical_crossentropy')

    @property
    def identifier(self):
        return f'{self.abbreviation}{int(self.use_event_attributes)}{int(self.use_case_attributes)}'

    def predict(self, inputs_f, inputs_b):
        out_f = self.net_f.predict(inputs_f)
        out_b = self.net_b.predict(inputs_b)
        if not isinstance(out_f, list):
            out_f = [out_f]
        if not isinstance(out_b, list):
            out_b = [out_b]
        return out_f, out_b

    def fit(self, dataset, batch_size=32, **kwargs):
        dataset.reverse(False)
        h1 = self.net_f.fit(dataset.features, dataset.targets, batch_size=batch_size, **kwargs)
        dataset.reverse(True)
        h2 = self.net_b.fit(dataset.features, dataset.targets, batch_size=batch_size, **kwargs)
        return h1, h2

    def save(self, file_name):
        self.net_f.save_weights(file_name + '_forward.h5')
        self.net_b.save_weights(file_name + '_backward.h5')

    def load(self, file_name):
        self.net_f([tf.ones(i) for i in ([(1, 1)] * len(self.net_f.rnn_inputs) + [(1,)] * len(self.net_f.fc_inputs))])
        self.net_f.load_weights(file_name + '_forward.h5')
        self.net_b([tf.ones(i) for i in ([(1, 1)] * len(self.net_b.rnn_inputs) + [(1,)] * len(self.net_b.fc_inputs))])
        self.net_b.load_weights(file_name + '_backward.h5')

    def batch_align(self, dataset, batch_size=5000, detailed=False, **kwargs):
        alignments = []
        start_beams = []
        start_probs = []
        beams = []
        probs = []
        costs = []

        for x, y in dataset.to_tf_dataset().batch(batch_size):
            if not isinstance(x, tuple):
                x = [x]
            a, b, c, sb, sp, p, _, _ = self.align([_x.numpy() for _x in x], detailed=True, **kwargs)

            alignments.append(a)
            start_beams.append(sb)
            start_probs.append(sp)
            beams.append(b)
            probs.append(p)
            costs.append(c)

        alignments = np.concatenate(alignments)
        start_beams = np.concatenate(start_beams)
        start_probs = np.concatenate(start_probs)
        beams = np.concatenate(beams)
        probs = np.concatenate(probs)
        costs = np.concatenate(costs)

        if detailed:
            return alignments, start_beams, start_probs, beams, probs, costs

        return alignments, beams, costs

    def align(self, dataset, k=5, hot_start=True, steps=10, delete_max=3, detailed=False):
        i = 0
        converged = False
        go_backwards = False
        start_probs = None

        if isinstance(dataset, Dataset):
            dataset.reverse(False)
            x = dataset.features
        else:
            x = dataset

        # Prepare data
        x_case = [_x for _x in x if len(_x.shape) == 1]
        x = [np.pad(_x, ((0, 0), (0, steps + 1))) for _x in x if len(_x.shape) == 2]  # Create space for inserts
        start_beams = np.copy(x[0])
        alive = np.ones(x[0].shape[0], dtype=bool)
        x_p = np.zeros(x[0].shape[0])

        # Alignments
        inserts = np.zeros_like(x[0])
        deletes = np.zeros_like(x[0])

        # Convergence
        last_beams_y = None

        for _ in range(steps):
            if converged:
                print('Converged')
                break

            # Keep time for progress output
            start_time = time()

            # Forwards data
            x_f = [_x[alive] for _x in x]
            y_f = [to_targets(_x) for _x in x_f]
            m_f = y_f[0] != 0

            # Backwards data
            reverse_mask = x[0][alive] != 0
            x_b = [reverse(_x[alive], reverse_mask) for _x in x]
            y_b = [to_targets(_x) for _x in x_b]
            m_b = y_b[0] != 0

            # RNN predictions
            _x_case = [_x[alive] for _x in x_case]
            y_pred_f, y_pred_b = self.predict(x_f + _x_case, x_b + _x_case)

            y_probs_f, cum_y_probs_f = log_probs(y_f[0], y_pred_f[0], m_f)
            y_probs_b, cum_y_probs_b = log_probs(y_b[0], y_pred_b[0], m_b)

            # Reverse backwards
            y_pred_b = [reverse(_y, reverse_mask) for _y in y_pred_b]
            cum_y_probs_b = reverse(cum_y_probs_b, reverse_mask)

            # Hot start
            if i == 0 and hot_start:
                scores = self.net_f.score(x_f, [np.copy(f) for f in y_pred_f])
                result = AnomalyDetectionResult(scores=scores, predictions=y_pred_f)
                b_f = Binarizer(result, ~m_f[:, :, None], np.dstack(x_f))
                detection_f = b_f.binarize(heuristic=Heuristic.LP_MEAN, strategy=Strategy.ATTRIBUTE)

            # Original probs
            if i == 0:
                start_probs = np.atleast_3d(cum_y_probs_f) + align(cum_y_probs_b, 1)
                start_probs = start_probs[:, :, 0].sum(-1) / ((~(x_f[0] == 0)).sum(-1) - 1)  # -1 to remove end symbol

            # BiBS step
            beams_p, beams_y, positions, p, y = bibs_step(x_f[0],
                                                          np.log(y_pred_f[0]), cum_y_probs_f,
                                                          np.log(y_pred_b[0]), cum_y_probs_b,
                                                          inserts[alive] > 0,
                                                          k=k, go_backwards=go_backwards, delete_max=delete_max)

            # Beams for event attributes
            beams_y = [beams_y]
            for n, (_y_f, _y_b) in enumerate(zip(y_pred_f[1:], y_pred_b[1:])):
                _y = (_y_f * align(_y_b, 1)).argmax(-1)
                _beams_y = gather(_y, positions - 1)
                _beams_y[beams_y[0] < 0] = beams_y[0][beams_y[0] < 0]
                beams_y.append(_beams_y)

            # Prepare old x
            if i == 0:
                # In the first run we have to repeat the original cases to match the dimension of `num_cases * k`
                x = [np.repeat(_x, k, 0) for _x in x]
                x_case = [np.repeat(_x, k) for _x in x_case]
                x_f = [np.repeat(_x, k, 0) for _x in x_f]
                x_p = np.repeat(x_p, k, 0)
                inserts = np.repeat(inserts, k, 0)
                deletes = np.repeat(deletes, k, 0)
                alive = np.repeat(alive, k, 0)
            else:
                # Get top-k beams for all cases. There are `k * k` beams available.
                shape = (alive.sum() // k, beams_p.shape[0] // (alive.sum() // k) * k)
                costs = (inserts[alive] > 0).sum(-1) + (deletes[alive] > 0).sum(-1)
                cost_y = np.zeros_like(beams_y[0])
                cost_y[beams_y[0] > 0] = 1
                cost_y[beams_y[0] < 0] = -beams_y[0][beams_y[0] < 0]
                cost_y[beams_y[0] == -42] = 0
                costs = costs[:, None] + cost_y

                idx = np.lexsort((-costs.reshape(shape), beams_p.reshape(shape)), axis=-1)[:, ::-1][:, :k]
                x_idx = (np.zeros_like(beams_p, dtype=int) + np.arange(alive.sum())[:, None]).reshape(shape)
                x_idx = gather(x_idx, idx).reshape(alive.sum())

                beams_y = [gather(_y.reshape(shape), idx) for _y in beams_y]
                positions = gather(positions.reshape(shape), idx)
                beams_p = gather(beams_p.reshape(shape), idx)
                x_f = [_x[x_idx] for _x in x_f]
                inserts[alive] = inserts[alive][x_idx]
                deletes[alive] = deletes[alive][x_idx]

            # Update probs
            x_p[alive] = beams_p.ravel()

            # New alignments
            inserts[alive], deletes[alive] = build_alignments(inserts[alive], deletes[alive],
                                                              beams_y[0], positions, i + 1)

            # Build new x
            for attr_i in range(len(x_f)):
                x[attr_i][alive] = build_beams(x_f[attr_i], np.copy(beams_y[attr_i]), positions)

            # Cases with all beams indicating 'do nothing' are finished
            finished = np.all(beams_y[0] == -42, -1)
            if i == 0 and hot_start:
                finished = np.logical_or(finished, np.all(detection_f[:, :, 0] == 0, -1))
            if last_beams_y is not None and beams_y[0].shape[0] == last_beams_y.shape[0]:
                finished = np.logical_or(finished, np.all(beams_y[0] == last_beams_y, -1))
            alive[alive] = np.repeat(~finished, k, 0)
            last_beams_y = beams_y[0]

            # Print progress
            print(
                f'Step {i + 1} {"←" if go_backwards else "→"} {time() - start_time}s {x[0].shape} finished={(~alive).sum() // k}')

            # Go the other way the next step
            go_backwards = not go_backwards

            # Converged
            converged = alive.sum() == 0

            # i++
            i += 1

        shape = (x[0].shape[0] // k, k, x[0].shape[1])
        beams = x[0].reshape(shape)
        inserts = inserts.reshape(shape)
        deletes = deletes.reshape(shape)
        costs = (inserts > 0).sum(-1) + (deletes > 0).sum(-1)
        probs = x_p.reshape((x[0].shape[0] // k, k))

        # Calculate alignments
        alignments = get_alignments(start_beams, beams, inserts, deletes)

        if detailed:
            return alignments, beams, costs, start_beams, start_probs, probs, inserts, deletes

        return alignments, beams, costs
