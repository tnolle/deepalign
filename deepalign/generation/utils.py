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

import itertools
from typing import Iterable

import numpy as np

from deepalign import EventLogGenerator
from deepalign.generation import AttributeAnomaly
from deepalign.generation import CategoricalAttributeGenerator
from deepalign.generation.example_values import company_names
from deepalign.generation.example_values import countries
from deepalign.generation.example_values import user_names
from deepalign.generation.example_values import week_days
from deepalign.generation.example_values import working_days
from deepalign.processmining import ProcessMap


def generate_for_process_model(process_model, size=5000, anomalies=None, anomaly_p=0.3, num_attr=0,
                               activity_dependency_ps=.25, attribute_dependency_ps=.75, p_var=5, seed=0, postfix='',
                               show_progress='tqdm'):
    if show_progress == 'tqdm':
        from tqdm import tqdm
    elif show_progress == 'tqdm_notebook':
        from tqdm import tqdm_notebook as tqdm

    if not isinstance(anomaly_p, Iterable):
        anomaly_p = [anomaly_p]

    if not isinstance(num_attr, Iterable):
        num_attr = [num_attr]

    if not isinstance(activity_dependency_ps, Iterable):
        activity_dependency_ps = [activity_dependency_ps]

    if not isinstance(attribute_dependency_ps, Iterable):
        attribute_dependency_ps = [attribute_dependency_ps]

    if not isinstance(p_var, Iterable):
        p_var = [p_var]

    process_map = ProcessMap.from_plg(process_model)

    event_attributes = [
        CategoricalAttributeGenerator(name='user', values=user_names, min_group=1, max_group=5),
        CategoricalAttributeGenerator(name='day', values=working_days, domain=week_days, min_group=2, max_group=5),
        CategoricalAttributeGenerator(name='country', values=countries, min_group=1, max_group=5),
        CategoricalAttributeGenerator(name='company', values=company_names, min_group=1, max_group=5)
    ]

    case_attributes = [
        CategoricalAttributeGenerator(name='decision', values=20, min_group=2, max_group=5),
        CategoricalAttributeGenerator(name='topic', values=10, min_group=2, max_group=8),
        CategoricalAttributeGenerator(name='quality', values=50, min_group=8, max_group=20),
        CategoricalAttributeGenerator(name='system', values=5, min_group=1, max_group=5)
    ]

    if anomalies is None:
        anomalies = []

    parameters = list(itertools.product(anomaly_p,
                                        num_attr,
                                        activity_dependency_ps,
                                        attribute_dependency_ps,
                                        p_var))

    np.random.seed(seed)
    seeds = np.random.randint(0, 10000, size=len(parameters))

    for i, (seed, params) in tqdm(enumerate(zip(seeds, parameters), start=1), desc=process_model, total=len(seeds)):
        anom_p, num_attr, act_dep_p, attr_dep_p, p_var = params

        _anomalies = anomalies
        if num_attr == 0:
            _anomalies = [a for a in anomalies if not isinstance(a, AttributeAnomaly)]

        # Save event log
        generator = EventLogGenerator(process_map,
                                      event_attributes=event_attributes[:num_attr],
                                      case_attributes=case_attributes[:num_attr])
        event_log = generator.generate(size=size,
                                       anomalies=_anomalies,
                                       anomaly_p=anom_p,
                                       activity_dependency_p=act_dep_p,
                                       attribute_dependency_p=attr_dep_p,
                                       probability_variance_max=p_var,
                                       seed=seed,
                                       show_progress=show_progress)

        generator.plot_likelihood_graph(f'graph_{process_model}{postfix}-{anom_p}-{i}.pdf', figsize=(20, 50))

        event_log.save(process_model + postfix, anom_p, i)
