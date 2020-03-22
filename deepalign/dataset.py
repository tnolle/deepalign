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

import json

import h5py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from deepalign.enums import AttributeType
from deepalign.enums import Class
from deepalign.enums import FeatureType
from deepalign.fs import ALIGNMENTS_DIR
from deepalign.fs import CORRECTIONS_DIR
from deepalign.fs import EventLogFile
from deepalign.generation import label_to_class
from deepalign.generation import prettify_label
from deepalign.processmining import Event
from deepalign.processmining.alignments import align
from deepalign.processmining.log import EventLog
from deepalign.utils import reverse

CASE_PREFIX = '[Case]_'


class Dataset(object):
    def __init__(self,
                 dataset_name=None,
                 use_case_attributes=False,
                 use_event_attributes=False,
                 go_backwards=False,
                 in_memory=True,
                 start=None,
                 end=None,
                 clip_len=None):

        # Parameters
        self.dataset_name = dataset_name
        self.use_case_attributes = use_case_attributes
        self.use_event_attributes = use_event_attributes
        self.in_memory = in_memory
        self.start = start if start is not None else 0
        self.end = end
        self.clip_len = clip_len

        # Properties
        self.case_lens = None
        self.classes = None
        self.labels = None
        self.encoders = None
        self.scalers = None
        self.case_encoders = None
        self._attribute_dims = None
        self._attribute_types = None
        self._attribute_keys = None
        self._feature_types = None
        self._event_log = None
        self._h5_file = None
        self._features = None
        self._event_features = None
        self._case_features = None
        self._correct_features = None

        # Load dataset
        if self.dataset_name is not None:
            self.load(self.dataset_name)

        if self.end is None:
            self.end = self.num_cases
        if self.clip_len is None:
            self.clip_len = self.max_len

        # Reverse
        if go_backwards:
            self.reverse(True)
        else:
            self.go_backwards = go_backwards

    def load(self, dataset_name):
        """
        Load dataset from disk. If there exists a cached file, load from cache. If no cache file exists, load from
        Event Log and cache it.

        :param dataset_name:
        :return:
        """
        # If features have already been lazy-loaded before, reset them
        if self._features is not None:
            del self._features
            self._features = None

        el_file = EventLogFile(dataset_name)
        self.dataset_name = el_file.name

        # Generate from event log
        if el_file.path.exists() and not el_file.cache_file.exists():
            self._event_log = EventLog.load(el_file.path)
            self.from_event_log(self._event_log)
            self._cache_dataset(el_file.cache_file)
            del self._features
            self._features = None

        # Always load from cache
        self._load_dataset_from_cache(el_file.cache_file)

    def _load_dataset_from_cache(self, file):
        with h5py.File(file, 'r') as file:
            self._attribute_dims = file.attrs['attribute_dims']
            self._attribute_keys = file.attrs['attribute_keys']
            self._attribute_types = file.attrs['attribute_types']
            self._feature_types = file.attrs['feature_types']

            self.encoders = {}
            self.scalers = {}
            for key, attr_type in zip(self.attribute_keys, self.attribute_types):
                features = file['features'][key]
                if 'encoder' in features.attrs and attr_type == AttributeType.CATEGORICAL:
                    enc = LabelEncoder()
                    enc.classes_ = np.array(features.attrs['encoder'])
                    self.encoders[key] = enc
                elif 'encoder_' + key in file['features']:
                    enc = LabelEncoder()
                    enc.classes_ = np.array([str(s, 'utf-8') for s in file['features']['encoder_' + key]])
                    self.encoders[key] = enc
                elif 'scaler' in features.attrs and attr_type == AttributeType.NUMERICAL:
                    scaler = StandardScaler()
                    mean, var, scale = features.attrs['scaler']
                    scaler.mean_ = mean
                    scaler.var_ = var
                    scaler.scale_ = scale
                    self.scalers[key] = scaler

            self.case_lens = np.array(file['case_lens'][self.start:self.end])
            if self.clip_len is not None:
                self.case_lens = np.clip(self.case_lens, a_min=None, a_max=self.clip_len)

            self.classes = np.array(file['classes'])
            self.labels = np.array([json.loads(l) for l in file['labels']])
            if file['labels'].shape != (0,):
                attr_clip = 1 if not self.use_event_attributes else None
                self.classes = self.classes[self.start:self.end, :, :attr_clip]
                self.labels = self.labels[self.start:self.end]

    def _cache_dataset(self, file):
        with h5py.File(file, 'w') as file:
            file.attrs['attribute_dims'] = self._attribute_dims
            file.attrs['attribute_keys'] = self._attribute_keys
            file.attrs['attribute_types'] = self._attribute_types
            file.attrs['feature_types'] = self._feature_types

            features = file.create_group('features')
            for x, key in zip(self._features, self._attribute_keys):
                dataset = features.create_dataset(key, data=x, compression="gzip", compression_opts=9)
                if key in self.encoders:
                    data = [c.encode('utf-8') for c in self.encoders[key].classes_]
                    features.create_dataset('encoder_' + key, data=data, compression='gzip', compression_opts=9)
                    # dataset.attrs['encoder'] = self.encoders[key].classes_.tolist()
                elif key in self.scalers:
                    scaler = self.scalers[key]
                    dataset.attrs['scaler'] = (scaler.mean_, scaler.var_, scaler.scale_)

            file.create_dataset('classes', data=self.classes, compression="gzip", compression_opts=9)
            file.create_dataset('case_lens', data=self.case_lens, compression="gzip", compression_opts=9)
            file.create_dataset('labels', data=np.array([np.string_(json.dumps(l)) for l in self.labels]),
                                compression="gzip", compression_opts=9)

    @property
    def h5_file(self):
        return EventLogFile(self.dataset_name).cache_file

    def reverse(self, go_backwards):
        if not self.in_memory and go_backwards != self.go_backwards:
            del self._features
            self._features = None
        self.go_backwards = go_backwards

    @property
    def _feature_indices(self):
        indices = []
        for i, feature_type in enumerate(self._feature_types):
            if feature_type == FeatureType.CONTROL_FLOW \
                    or feature_type == FeatureType.EVENT and self.use_event_attributes \
                    or feature_type == FeatureType.CASE and self.use_case_attributes:
                indices.append(i)
        return indices

    @property
    def _target_indices(self):
        indices = []
        for i, feature_type in enumerate(self._feature_types):
            if feature_type == FeatureType.CONTROL_FLOW \
                    or feature_type == FeatureType.EVENT and self.use_event_attributes:
                indices.append(i)
        return indices

    @property
    def feature_types(self):
        return self._feature_types[self._feature_indices]

    @property
    def attribute_dims(self):
        return self._attribute_dims[self._feature_indices]

    @property
    def attribute_keys(self):
        return self._attribute_keys[self._feature_indices]

    @property
    def attribute_types(self):
        return self._attribute_types[self._feature_indices]

    @property
    def target_dims(self):
        return self.attribute_dims[self._target_indices]

    @property
    def target_keys(self):
        return self.attribute_keys[self._target_indices]

    @property
    def target_types(self):
        return self.attribute_types[self._target_indices]

    @property
    def mask(self):
        """Return boolean mask where padding is False"""
        return self.targets[0] != 0

    @property
    def event_log(self):
        """Return the event log object of this dataset."""
        if self.dataset_name is None:
            raise ValueError(f'dataset {self.dataset_name} cannot be found')

        if self._event_log is None:
            self._event_log = EventLog.load(self.dataset_name)
        return self._event_log

    @property
    def binary_classes(self):
        """Return targets for anomaly detection; 0 = normal, 1 = anomaly."""
        if self.classes is not None and len(self.classes) > 0:
            return (self.classes > Class.ANOMALY).astype(int)
        return None

    @property
    def features(self):
        if self.in_memory:
            if self._features is None:
                with h5py.File(self.h5_file, 'r') as file:
                    self._features = []
                    for k, t in zip(self.attribute_keys, self.feature_types):
                        if t == FeatureType.CONTROL_FLOW or t == FeatureType.EVENT:
                            f = np.array(file['features'][k][self.start:self.end][:, :self.clip_len])
                        else:
                            f = np.array(file['features'][k][self.start:self.end])
                        self._features.append(f)
            if self.go_backwards:
                return [reverse(f) for f in self._features]

        if not self.in_memory and self._features is None:
            def normalize(f):
                if f.ndim == 1:
                    f = f[:self.clip_len]
                    if self.go_backwards:
                        f = reverse(f)
                    return f
                else:
                    f = f[:, :self.clip_len]
                    if self.go_backwards:
                        f = reverse(f)
                    return f

            self._features = [tf.keras.utils.HDF5Matrix(self.h5_file, f'features/{k}', start=self.start, end=self.end,
                                                        normalizer=normalize) for k in self.attribute_keys]

        return self._features

    @property
    def flat_features(self):
        """
        Return combined features in one single tensor.

        `features` returns one tensor per attribute. This method combines all attributes into one tensor. Resulting
        shape of the tensor will be (number_of_cases, max_case_length, number_of_attributes).

        :return:
        """
        return np.dstack(self.features)

    @staticmethod
    def onehot(x):
        return np.eye(np.max(x) + 1, dtype=int)[x.astype(int)]

    @property
    def onehot_features(self):
        """
        Return one-hot encoding of integer encoded features.

        As `features` this will return one tensor for each attribute. Shape of tensor for each attribute will be
        (number_of_cases, max_case_length, attribute_dimension). The attribute dimension refers to the number of unique
        values of the respective attribute encountered in the event log.

        :return:
        """
        return [self.onehot(f)[:, :, 1:] for f in self.features]

    @property
    def flat_onehot_features(self):
        """
        Return combined one-hot features in one single tensor.

        One-hot vectors for each attribute in each event will be concatenated. Resulting shape of tensor will be
        (number_of_cases, max_case_length, attribute_dimension[0] + attribute_dimension[1] + ... + attribute_dimension[n]).

        :return:
        """
        return np.concatenate(self.onehot_features, axis=2)

    @staticmethod
    def remove_time_dimension(x):
        return x.reshape((x.shape[0], np.product(x.shape[1:])))

    @property
    def flat_features_2d(self):
        """
        Return 2d tensor of flat features.

        Concatenates all attributes together, removing the time dimension. Resulting tensor shape will be
        (number_of_cases, max_case_length * number_of_attributes).

        :return:
        """
        return self.remove_time_dimension(self.flat_features)

    @property
    def flat_onehot_features_2d(self):
        """
        Return 2d tensor of one-hot encoded features.

        Same as `flat_onehot_features`, but with flattened time dimension (the second dimension). Resulting tensor shape
        will be (number_of_cases, max_case_length * (attribute_dimension[0] + attribute_dimension[1] + ... + attribute_dimension[n]).

        :return:
        """
        return self.remove_time_dimension(self.flat_onehot_features)

    @property
    def targets(self):
        """
        Return targets to be used when training predictive anomaly detectors.

        Returns for each case the case shifted by one event to the left. A predictive anomaly detector is trained to
        predict the nth + 1 event of a case when given the first n events.

        :return:
        """

        def normalize(f):
            if f.ndim == 1:
                t = f[:self.clip_len]
                if self.go_backwards:
                    t = reverse(t)
                t = np.pad(t[1:], (0, 1), mode='constant')
                if self.go_backwards:
                    t = (reverse(t) > 0).astype(int) * t
                return t
            else:
                t = f[:, :self.clip_len]
                if self.go_backwards and not self.in_memory:
                    t = reverse(t)
                t = np.pad(t[:, 1:], ((0, 0), (0, 1)), mode='constant')
                if self.go_backwards:
                    if self.in_memory:
                        t = (t > 0).astype(int) * t
                    else:
                        t = (reverse(t) > 0).astype(int) * t
                return t

        if self.in_memory:
            return [normalize(f) for f in [self.features[i] for i in self._target_indices]]
        else:
            return [tf.keras.utils.HDF5Matrix(self.h5_file, f'features/{k}', start=self.start, end=self.end,
                                              normalizer=normalize) for k in self.target_keys]

    def to_tf_dataset(self):
        features = tuple(tf.data.Dataset.from_tensor_slices(f[:]) for f in self.features)
        targets = tuple(tf.data.Dataset.from_tensor_slices(t[:]) for t in self.targets)
        features = tf.data.Dataset.zip(features) if len(features) > 1 else features[0]
        targets = tf.data.Dataset.zip(targets) if len(targets) > 1 else targets[0]
        tf_dataset = tf.data.Dataset.zip((features, targets))
        return tf_dataset

    @property
    def alignments(self):
        if (ALIGNMENTS_DIR / (self.dataset_name + '.h5')).exists():
            with h5py.File(str(ALIGNMENTS_DIR / (self.dataset_name + '.h5')), 'r') as file:
                alignments = np.array(file['alignments'])
                costs = np.array(file['costs'])
        else:
            _alignments = []
            costs = []
            for i in tqdm(range(self.num_cases), total=self.num_cases):
                x = self.features[0][i]
                y = self.correct_features[0][i]
                alignment = align(x, y)[0]
                _alignments.append(alignment)
                cost = (alignment == 0).sum()
                costs.append(cost)
            max_len = np.max([a.shape[1] for a in _alignments])
            alignments = -np.ones((len(_alignments), 2, max_len), dtype=int)
            for i, a in enumerate(_alignments):
                alignments[i, :, :a.shape[1]] = a
            costs = np.array(costs)
            with h5py.File(str(ALIGNMENTS_DIR / (self.dataset_name + '.h5')), 'w') as file:
                file.create_dataset('alignments', data=alignments, compression="gzip", compression_opts=9)
                file.create_dataset('costs', data=costs, compression="gzip", compression_opts=9)
        return alignments, costs

    @property
    def correct_features(self):
        if self._correct_features is None:
            if (CORRECTIONS_DIR / (self.dataset_name + '.h5')).exists():
                self._correct_features = []
                with h5py.File(str(CORRECTIONS_DIR / (self.dataset_name + '.h5')), 'r') as file:
                    self._correct_features.append(np.array(file['features']['name']))
            else:
                # If it is a generated one, recreate the original features by reversing the anomaly operations
                encoders = {}
                for key, encoder in self.encoders.items():
                    encoders[key] = LabelEncoder()
                    if not key.startswith('[Case]_'):
                        encoders[key].classes_ = encoder.classes_[1:]
                correct_features = [np.copy(self.features[i]) for i in self._target_indices]
                for i, label in tqdm(enumerate(self.labels), total=self.num_cases):
                    if label != 'normal':
                        anomaly = label['anomaly']
                        if anomaly == 'Insert':
                            indices = [idx + 1 for idx in label['attr']['indices']]
                            reorder = [idx for idx in range(self.max_len) if idx not in indices] + indices
                            for k in range(len(correct_features)):
                                correct_features[k][i][indices] = 0
                                correct_features[k][i] = correct_features[k][i][reorder]
                        elif anomaly == 'Attribute':
                            keys = label['attr']['attribute']
                            attr_indices = label['attr']['attribute_index']
                            indices = label['attr']['index']
                            original = label['attr']['original']
                            for key, j, k, v in zip(keys, attr_indices, indices, original):
                                if key in encoders:
                                    try:
                                        correct_features[j + 1][i][k + 1] = encoders[key].transform([v]) + 1
                                    except:
                                        pass
                        elif anomaly == 'Rework':
                            size = label['attr']['size']
                            start = label['attr']['start'] + 1
                            indices = list(range(start, start + size, 1))
                            reorder = [idx for idx in range(self.max_len) if idx not in indices] + indices
                            for k in range(len(correct_features)):
                                correct_features[k][i][start:start + size] = 0
                                correct_features[k][i] = correct_features[k][i][reorder]
                        elif anomaly == 'SkipSequence':
                            start = label['attr']['start'] + 1
                            size = label['attr']['size']
                            indices = list(range(start, start + size, 1))
                            reorder = list(range(0, self.max_len - size, 1))
                            reorder = reorder[:start] + [-1] * size + reorder[start:]
                            for k in range(len(correct_features)):
                                key = self.attribute_keys[k]
                                correct_features[k][i] = correct_features[k][i][reorder]
                                skipped = [
                                    e['name'] if key == 'name' else e['attributes'].get(key.replace('_', ':'), '▶') for
                                    e
                                    in label['attr']['skipped']]
                                if key in encoders:
                                    try:
                                        correct_features[k][i][indices] = encoders[key].transform(skipped) + 1
                                    except:
                                        pass
                        elif anomaly == 'Early':
                            size = label['attr']['size']
                            shift_from = label['attr']['shift_from'] + 1
                            shift_to = label['attr']['shift_to'] + 1
                            reorder = list(range(self.max_len))
                            shifted = reorder[shift_to:shift_to + size]
                            reorder = reorder[:shift_to] + reorder[shift_to + size:shift_from] + shifted + reorder[
                                                                                                           shift_from:]
                            for k in range(len(correct_features)):
                                correct_features[k][i] = correct_features[k][i][reorder]
                        elif anomaly == 'Late':
                            size = label['attr']['size']
                            shift_from = label['attr']['shift_from'] + 1
                            shift_to = label['attr']['shift_to'] + 1
                            distance = shift_to - shift_from
                            reorder = list(range(self.max_len))
                            shifted = reorder[shift_to:shift_to + size]
                            reorder = \
                                reorder[:shift_from] + shifted + \
                                reorder[shift_from:shift_from + distance] + reorder[shift_to + size:]
                            for k in range(len(correct_features)):
                                correct_features[k][i] = correct_features[k][i][reorder]
                self._correct_features = correct_features

                with h5py.File(str(CORRECTIONS_DIR / (self.dataset_name + '.h5')), 'w') as file:
                    features = file.create_group('features')
                    for x, key in zip(self._correct_features, self._attribute_keys):
                        features.create_dataset(key, data=x, compression="gzip", compression_opts=9)

        return self._correct_features

    @property
    def pretty_labels(self):
        return np.array([prettify_label(l) for l in self.labels])

    @property
    def text_labels(self):
        """Return the labels transformed into text, one string for each case in the event log."""
        return np.array(['Normal' if label == 'normal' else label['anomaly'] for label in self.labels])

    @property
    def unique_text_labels(self):
        """Return unique text labels."""
        return sorted(set(self.text_labels))

    @property
    def unique_anomaly_text_labels(self):
        """Return only the unique anomaly text labels."""
        return [label for label in self.unique_text_labels if label != 'Normal']

    def get_indices_for_type(self, t):
        if len(self.text_labels) > 0:
            return np.where(self.text_labels == t)[0]
        else:
            return range(int(self.num_cases))

    @property
    def normal_indices(self):
        return self.get_indices_for_type('Normal')

    @property
    def cf_anomaly_indices(self):
        if len(self.text_labels) > 0:
            return np.where(np.logical_and(self.text_labels != 'Normal', self.text_labels != 'Attribute'))[0]
        else:
            return range(int(self.num_cases))

    @property
    def anomaly_indices(self):
        if len(self.text_labels) > 0:
            return np.where(self.text_labels != 'Normal')[0]
        else:
            return range(int(self.num_cases))

    @property
    def num_attributes(self):
        """Return the number of attributes in the event log."""
        return self.num_event_attributes + self.num_case_attributes

    @property
    def num_event_attributes(self):
        return len([t for t in self.feature_types if t == FeatureType.CONTROL_FLOW or t == FeatureType.EVENT])

    @property
    def num_case_attributes(self):
        return len([t for t in self.feature_types if t == FeatureType.CASE])

    @property
    def num_cases(self):
        """Return number of cases in the event log, i.e., the number of examples in the dataset."""
        return len(self.case_lens)

    @property
    def num_events(self):
        """Return the total number of events in the event log."""
        return sum(self.case_lens)

    @property
    def max_len(self):
        """Return the length of the case with the most events."""
        return max(self.case_lens)

    @staticmethod
    def _get_classes_and_labels_from_event_log(event_log):
        """
        Extract anomaly labels from event log format and transform into anomaly detection classes.

        :param event_log:
        :return:
        """
        labels = np.asarray([case.attributes['label'] for case in event_log if
                             case.attributes is not None and 'label' in case.attributes])

        # +2 for start and end event
        num_events = event_log.max_case_len + 2
        num_attributes = event_log.num_event_attributes
        classes = np.asarray([label_to_class(label, num_events, num_attributes) for label in labels])

        return classes, labels

    @staticmethod
    def _from_event_log(event_log, event_attrs=None, case_attrs=None):
        """
        Transform event log as feature columns.

        Categorical attributes are integer encoded. Shape of feature columns is
        (number_of_cases, max_case_length, number_of_attributes).
        """

        # Event attributes to include
        if event_attrs is None:
            event_attrs = event_log.event_attribute_keys
        event_attr_types = event_log.get_event_attribute_types(event_attrs)

        # Case attributes to include
        if case_attrs is None:
            case_attrs = event_log.case_attribute_keys
        case_attr_types = event_log.get_case_attribute_types(case_attrs)

        # Feature columns
        event_feature_columns = dict(name=[])
        case_feature_columns = {}
        case_lens = []

        # Remove numerical attributes
        # event_attrs = [a for a, t in zip(event_attrs, event_attr_types) if t == AttributeType.CATEGORICAL]
        # event_attr_types = [t for t in event_attr_types if t == AttributeType.CATEGORICAL]

        # Create beginning of sequence event
        start_event = dict((a, EventLog.start_symbol if t == AttributeType.CATEGORICAL else 0.0) for a, t in
                           zip(event_attrs, event_attr_types))
        start_event = Event(timestamp=None, **start_event)

        # Create end of sequence event
        end_event = dict((a, EventLog.end_symbol if t == AttributeType.CATEGORICAL else 0.0) for a, t in
                         zip(event_attrs, event_attr_types))
        end_event = Event(timestamp=None, **end_event)

        # Save all values in a flat 1d array. We will reshape it later.
        for i, case in enumerate(event_log.cases):
            case_lens.append(case.num_events + 2)  # +2 for start and end events

            for key, attr_type in zip(case_attrs, case_attr_types):
                if key in case.attributes:
                    attr = case.attributes[key]
                elif attr_type == AttributeType.CATEGORICAL:
                    attr = '<Blank>'
                else:
                    attr = 0.0

                if key not in case_feature_columns.keys():
                    case_feature_columns[key] = []
                case_feature_columns[key].append(attr)

            for event in [start_event] + case.events + [end_event]:
                for key, attr_type in zip(event_attrs, event_attr_types):
                    if key == 'name':
                        attr = event.name
                    elif key in event.attributes:
                        attr = event.attributes[key]
                    elif attr_type == AttributeType.CATEGORICAL:
                        attr = '<Blank>'
                    else:
                        attr = 0.0

                    if key not in event_feature_columns.keys():
                        event_feature_columns[key] = []
                    event_feature_columns[key].append(attr)

        case_encoders = {}
        case_scalers = {}
        for key, attr_type in zip(case_attrs, case_attr_types):
            if attr_type == AttributeType.CATEGORICAL:
                encoder = LabelEncoder()
                case_feature_columns[key] = encoder.fit_transform(case_feature_columns[key])
                case_encoders[key.replace(':', '_').replace(' ', '_')] = encoder
            elif attr_type == AttributeType.NUMERICAL:
                scaler = StandardScaler()
                case_feature_columns[key] = scaler.fit_transform(np.array(case_feature_columns[key]).reshape(-1, 1))[:,
                                            0]
                case_scalers[key.replace(':', '_').replace(' ', '_')] = scaler

        # Data preprocessing
        event_encoders = {}
        event_scalers = {}
        for key, attr_type in zip(event_attrs, event_attr_types):
            if attr_type == AttributeType.CATEGORICAL:
                encoder = LabelEncoder()
                event_feature_columns[key] = encoder.fit_transform(event_feature_columns[key]) + 1
                encoder.classes_ = np.concatenate((['•'], encoder.classes_))  # Add padding at position 0
                event_encoders[key.replace(':', '_').replace(' ', '_')] = encoder
            elif attr_type == AttributeType.NUMERICAL:
                scaler = StandardScaler()
                event_feature_columns[key] = scaler.fit_transform(np.array(event_feature_columns[key]).reshape(-1, 1))[
                                             :, 0]
                event_scalers[key.replace(':', '_').replace(' ', '_')] = scaler

        # Retrieve case level features
        case_features = [case_feature_columns[key] for key in case_attrs]

        # Transform event features back into sequences
        dtypes = [int if t == AttributeType.CATEGORICAL else float for t in event_attr_types]
        case_lens = np.array(case_lens)
        offsets = np.concatenate(([0], np.cumsum(case_lens)[:-1]))
        event_features = [np.zeros((case_lens.shape[0], case_lens.max()), dtype=dtype) for dtype in dtypes]
        for i, (offset, case_len) in enumerate(zip(offsets, case_lens)):
            for k, key in enumerate(event_feature_columns):
                event_features[k][i, :case_len] = event_feature_columns[key][offset:offset + case_len]

        # Replace illegal characters for layer names
        event_attr_keys = [a.replace(':', '_').replace(' ', '_') for a in event_attrs]
        case_attr_keys = [a.replace(':', '_').replace(' ', '_') for a in case_attrs]

        # Prepare return values
        event_data = (event_features, event_attr_types, event_attr_keys, event_encoders, event_scalers)
        case_data = (case_features, case_attr_types, case_attr_keys, case_encoders, case_scalers)

        return event_data, case_data, case_lens

    def from_event_log(self, event_log):
        """
        Load event log file and set the basic fields of the `Dataset` class.

        :param event_log: event log name as string
        :return:
        """
        # Get features from event log
        event_data, case_data, self.case_lens = self._from_event_log(event_log)
        self._features, self._attribute_types, self._attribute_keys, self.encoders, self.scalers = event_data
        event_features, event_attr_types, event_attr_keys, event_encoders, event_scalers = event_data
        case_features, case_attr_types, case_attr_keys, case_encoders, case_scalers = case_data

        # Combine
        self._features = event_features + case_features
        self._attribute_keys = event_attr_keys + [f'{CASE_PREFIX}{a}' for a in case_attr_keys]
        self._attribute_types = event_attr_types + case_attr_types
        self._feature_types = [FeatureType.CONTROL_FLOW] + \
                              [FeatureType.EVENT] * (len(event_features) - 1) + \
                              [FeatureType.CASE] * len(case_features)
        self.encoders = dict(**event_encoders, **dict((f'{CASE_PREFIX}{k}', v) for k, v in case_encoders.items()))
        self.scalers = dict(**event_scalers, **dict((f'{CASE_PREFIX}{k}', v) for k, v in case_scalers.items()))

        # Calculate dimensions
        self._attribute_dims = [f.max() if t == AttributeType.CATEGORICAL else 1
                                for f, t in zip(self._features, self._attribute_types)]

        # Get targets and labels from event log
        self.classes, self.labels = self._get_classes_and_labels_from_event_log(event_log)
