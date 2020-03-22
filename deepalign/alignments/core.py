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

import pickle

from deepalign.fs import AlignerFile


class Aligner:
    abbreviation = None
    name = None

    def __init__(self, model=None):
        self.model = None
        if model is not None:
            self.load(model)

    def save(self, file_name=None):
        """Save the class instance using pickle.

        :param file_name: Custom file name
        :return: the file path
        """
        if self.model is not None:
            model_file = AlignerFile(file_name)
            with open(model_file.str_path, 'wb') as f:
                pickle.dump(self.model, f)
            return model_file
        else:
            raise RuntimeError(
                'Saving not possible. No model has been trained yet.')

    def load(self, file_name):
        # load model file
        model_file = AlignerFile(file_name)

        # load model
        self.model = pickle.load(open(model_file.path, 'rb'))

    def fit(self, dataset):
        raise NotImplementedError()

    def align(self, dataset):
        raise NotImplementedError()
