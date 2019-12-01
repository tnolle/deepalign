#  Copyright 2019 Timo Nolle
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

import subprocess
from multiprocessing import Pool

import editdistance
import numpy as np
import pm4pycvxopt
from pm4py.algo.conformance.alignments import factory as aligner
from pm4py.algo.discovery.alpha import factory as alpha_miner
from pm4py.algo.discovery.dfg import factory as dfg_miner
from pm4py.algo.discovery.heuristics import factory as heuristics_miner
from pm4py.algo.discovery.inductive import factory as inductive_miner
from pm4py.objects.conversion.dfg import factory as dfg_mining_factory
from pm4py.objects.log.log import Event
from pm4py.objects.log.log import EventLog
from pm4py.objects.log.log import Trace
from pm4py.objects.petri.importer import factory as pnml_importer
from tqdm import tqdm

from deepalign import fs
from deepalign.alignments.core import Aligner
from deepalign.processmining.alignments import needleman_wunsch


def align_top_k_cost(obj):
    case, paths, k = obj
    costs = np.array([editdistance.eval(case, path) for path in paths])
    best_indices = np.argsort(costs)[:k]
    return [needleman_wunsch(case, paths[i])[0] for i in best_indices]


class OptimalCostAligner(Aligner):
    abbreviation = 'optimal'
    name = 'OptimalCost'

    def __init__(self, model=None):
        super(OptimalCostAligner, self).__init__(model=model)

    def fit(self, dataset):
        self.model = [np.trim_zeros(f).tolist() for f in np.unique(dataset.correct_features[0], axis=0)]

    def align(self, dataset, k=5):
        cases, index, inverse = np.unique(dataset.features[0], return_index=True, return_inverse=True, axis=0)
        k = min(len(self.model), k)

        _alignments = []
        with Pool() as pool:
            for alignment in tqdm(pool.imap(align_top_k_cost, [(c, self.model, k) for c in cases]),
                                  total=len(cases), desc=dataset.dataset_name):
                _alignments.append(alignment)

        max_len = max([len(a[0]) for alignment in _alignments for a in alignment])

        costs = np.zeros((len(cases), k))
        alignments = -np.ones((len(cases), k, 2, max_len))
        beams = np.zeros(((len(cases)), k, max_len))
        for i, alignment in enumerate(_alignments):
            for j, a in enumerate(alignment):
                alignments[i, j, :, :a.shape[1]] = a
                beam = np.array([_a for _a in a[1] if _a != 0])
                beams[i, j, :beam.shape[0]] = beam
                costs[i, j] = (a == 0).sum()

        return alignments[inverse], beams[inverse], costs[inverse]


def pm4py_align(obj):
    trace, net, im, fm = obj
    return aligner.apply_log([trace], net, im, fm)[0]


class PM4PYAligner(Aligner):
    miner = None
    parameters = None
    fast = pm4pycvxopt  # ensure import is not being removed

    def __init__(self, model=None):
        super(PM4PYAligner, self).__init__(model=model)

    def _convert_log(self, dataset):
        _, index, inverse = np.unique(dataset.features[0], return_index=True, return_inverse=True, axis=0)

        log = EventLog()
        for case in dataset.event_log[index]:
            trace = Trace()
            for e in case:
                event = Event()
                event['concept:name'] = e.name
                trace.append(event)
            log.append(trace)

        return log, inverse

    def fit(self, dataset):
        log, inverse = self._convert_log(dataset)
        self.model = self.miner.apply([log[i] for i in inverse], parameters=self.parameters)

    def align(self, dataset):
        log, inverse = self._convert_log(dataset)
        net, im, fm = self.model

        _alignments = []
        with Pool() as pool:
            for a in tqdm(pool.imap(pm4py_align, [(trace, net, im, fm) for trace in log]), total=len(log)):
                _alignments.append(a)
        # _alignments = [aligner.apply_log([trace], net, im, fm)[0] for trace in tqdm(log)]

        _alignments = [[a for a in alignment['alignment'] if a != ('>>', None)] for alignment in _alignments]
        max_len = max([len(alignment) for alignment in _alignments]) + 2  # +2 for start and end symbol

        start_symbol = dataset.attribute_dims[0]
        end_symbol = dataset.attribute_dims[0] - 1

        encode = dict((c, i) for i, c in enumerate(dataset.encoders['name'].classes_))
        encode['>>'] = 0

        costs = np.zeros((len(_alignments), 1))
        alignments = -np.ones((len(_alignments), 1, 2, max_len))
        beams = np.zeros((len(_alignments), 1, max_len), dtype=int)

        for i, alignment in enumerate(_alignments):
            alignment = np.array(
                [[start_symbol, start_symbol]] +
                [[encode[a[0]], encode[a[1]]] for a in alignment] +
                [[end_symbol, end_symbol]]
            ).T
            alignments[i, :, :, :alignment.shape[1]] = alignment
            beam = np.array([a for a in alignment[1] if a != 0])
            beams[i, :, :beam.shape[0]] = beam
            costs[i] = (alignment == 0).sum()

        return alignments[inverse], beams[inverse], costs[inverse]


class AlphaMinerAligner(PM4PYAligner):
    abbreviation = 'alpha'
    name = 'AlphaMiner'

    miner = alpha_miner

    def __init__(self, model=None):
        super(AlphaMinerAligner, self).__init__(model=model)


class AlphaMinerPlusAligner(PM4PYAligner):
    abbreviation = 'alphaplus'
    name = 'AlphaMinerPlus'

    miner = alpha_miner

    def __init__(self, model=None):
        super(AlphaMinerPlusAligner, self).__init__(model=model)

    def fit(self, dataset):
        log, inverse = self._convert_log(dataset)
        self.model = self.miner.apply([log[i] for i in inverse], variant='plus')


class HeuristicsMinerAligner(PM4PYAligner):
    abbreviation = 'hm'
    name = 'HeuristicsMiner'

    miner = heuristics_miner
    parameters = {'dependency_thresh': 0.99}

    def __init__(self, model=None):
        super(HeuristicsMinerAligner, self).__init__(model=model)


class InductiveMinerAligner(PM4PYAligner):
    abbreviation = 'im'
    name = 'InductiveMiner'

    miner = inductive_miner

    def __init__(self, model=None):
        super(InductiveMinerAligner, self).__init__(model=model)


class DFGMinerAligner(PM4PYAligner):
    abbreviation = 'dfg'
    name = 'DFGMiner'

    miner = dfg_miner

    def __init__(self, model=None):
        super(DFGMinerAligner, self).__init__(model=model)

    def fit(self, dataset):
        log, inverse = self._convert_log(dataset)
        dfg = self.miner.apply([log[i] for i in inverse])
        self.model = dfg_mining_factory.apply(dfg)


class SplitMinerAligner(PM4PYAligner):
    abbreviation = 'sm'
    name = 'SplitMiner'

    def fit(self, dataset):
        el_path = str(fs.OUT_DIR / 'xes' / (dataset.dataset_name + '.xes'))
        splitminer_dir = fs.RES_DIR / 'splitminer'
        splitminer = str(splitminer_dir / 'splitminer.jar')
        out_path = str(splitminer_dir / 'outputs' / dataset.dataset_name)
        lib = str(splitminer_dir / 'lib')

        subprocess.call(
            [f'java', f'-cp', f'{splitminer}:{lib}/*', 'au.edu.unimelb.services.ServiceProvider', 'SMPN', '0.0', '0.4',
             'true', el_path, out_path])

        self.model = pnml_importer.apply(out_path + '.pnml')
