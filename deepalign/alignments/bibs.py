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

import numpy as np

from deepalign.utils import align
from deepalign.utils import gather


def top_k(p, y, k=5):
    positions = np.zeros_like(p, dtype=int) + np.arange(p.shape[1])[None, :, None] + 1

    shape = (p.shape[0], np.product(p.shape[1:]))
    idx = p.reshape(shape).argsort(-1)[:, ::-1][:, :k]

    p_new = gather(p.reshape(shape), idx)
    y_new = gather(y.reshape(shape), idx)
    positions = gather(positions.reshape(shape), idx)

    return p_new, y_new, positions


def bibs_step(x, head_preds, head_probs, tail_preds, tail_probs, guard=None, k=5, go_backwards=False, delete_max=3):
    # Top-k prections for every timestamp
    if not go_backwards:
        y = head_preds.argsort(-1)[:, :, ::-1][:, :, :k]
    else:
        y = align(tail_preds.argsort(-1)[:, :, ::-1][:, :, :k], 1)

    mask = x == 0
    y[mask] = 0

    p_empty = np.atleast_3d(head_probs) + align(tail_probs, 1)
    p_empty = p_empty[:, :, 0].sum(-1) / ((~mask).sum(-1) - 1)  # -1 to remove end symbol

    p_y = \
        align(head_probs, -1) + \
        gather(head_preds, y) + \
        gather(align(tail_preds, 1), y) + \
        align(tail_probs, 2)
    p_y[align(mask, 1, 1)[:, :, 0]] = -np.inf

    def p_remove_next(i):
        p_remove = \
            align(head_probs, -1) + \
            gather(head_preds, align(x, 1 + i)) + \
            gather(align(tail_preds, 1 + i), np.atleast_3d(x)) + \
            align(tail_probs, 2 + i)
        p_remove[align(mask, 1 + i, 1)] = -np.inf
        if guard is not None:
            for j in range(i):
                p_remove[align(guard, j + 1, 0)] = -np.inf
        return p_remove

    p_remove = np.concatenate([p_remove_next(i + 1) for i in range(delete_max)], -1)
    y_remove = \
        np.zeros((y.shape[0], y.shape[1], p_remove.shape[-1]), dtype=int) + \
        np.array([-(i + 1) for i in range(delete_max)])[None, None, :]

    # Combine
    p = np.concatenate((p_y, p_remove), -1)
    y = np.concatenate((y, y_remove), -1)

    # Mask
    p[mask] = -np.inf
    y[align(mask, 1, 1)[:, :, 0]] = 0

    # Insert empty at bottom right, this will always be free, and this makes the code much simpler
    p[:, -1, -1] = p_empty
    y[:, -1, -1] = -42  # Identifier for 'do nothing'

    # Top-k beams
    beams_p, beams, positions = top_k(p, y, k=k)

    return beams_p, beams, positions, p, y


def get_indices(indices, types, l):
    idx = np.zeros((len(indices), l), dtype=int)
    for j, (i, t) in enumerate(zip(indices, types)):
        normal = np.arange(l)
        if t == -42 or t == 0 or i == 0:
            idx[j, :] = normal
        elif t >= 0:
            idx[j, :i] = normal[:i]
            idx[j, i] = l - 1
            idx[j, i + 1:] = normal[i:-1]
        elif t < 0:
            idx[j, :i] = normal[:i]
            idx[j, i:t] = normal[i - t:]
            idx[j, t:] = normal[-1]
    return idx


def build_beams(x, y, pos):
    idx = get_indices(pos.ravel(), y.ravel(), l=x.shape[1])
    y[y < 0] = 0
    x[:, -1] = y.ravel()
    return gather(x, idx)


def get_delete_indices(indices, types, l):
    idx = np.zeros((len(indices), l), dtype=int)
    for j, (i, t) in enumerate(zip(indices, types)):
        normal = np.arange(l)
        if t == -42 or t >= 0 or i == 0:
            idx[j, :] = normal
        elif t < 0:
            idx[j, :i] = normal[:i]
            idx[j, i:i - t] = l - 1
            idx[j, i - t:] = normal[i - t:]
    return idx


def build_alignments(inserts, deletes, y, pos, step):
    for j, (i, t) in enumerate(zip(pos.ravel(), y.ravel())):
        if t == -42 or i == 0 or t > 0:
            continue
        insert_offset = (inserts[j, :i] > 0).sum()
        delete_offset = (deletes[j, :i] > 0).sum()
        d = -t
        for k in range(deletes.shape[1] - i):
            if d == 0:
                break
            if deletes[j, i - insert_offset + delete_offset + k] == 0:
                deletes[j, i - insert_offset + delete_offset + k] = step
                d -= 1

    insert_idx = get_indices(pos.ravel(), y.ravel(), l=inserts.shape[1])
    insert_y = np.copy(y)
    insert_y[y == -42] = 0
    insert_y[y < 0] = 0
    insert_y[y > 0] = step
    inserts[:, -1] = insert_y.ravel()
    inserts = gather(inserts, insert_idx)

    return inserts, deletes


def get_alignment(log, model, inserts, deletes):
    if np.all(log == model):
        log = log[log != 0]
        model = model[model != 0]
    else:
        log = log.tolist()
        model = model.tolist()
        inserts = inserts.tolist()
        deletes = deletes.tolist()

        end = len(log)
        for i in range(len(log)):
            if log[i] == model[i] == 0:
                end = i
                break
            if deletes[i] > 0:
                model = model[:i] + [0] + model[i:]
                inserts = inserts[:i] + [0] + inserts[i:]
            if inserts[i] > 0:
                log = log[:i] + [0] + log[i:]
                deletes = deletes[:i] + [0] + deletes[i:]

        log = log[:end]
        model = model[:end]

    alignment = np.vstack((log, model))

    return alignment


def get_alignments(originals, beams, inserts, deletes):
    alignments = -np.ones((*beams.shape[:-1], 2, beams.shape[-1]), dtype=int)
    for case_index in range(originals.shape[0]):
        l = originals[case_index]
        for beam_index in range(beams.shape[1]):
            m = beams[case_index][beam_index]
            i = inserts[case_index][beam_index]
            d = deletes[case_index][beam_index]
            a = get_alignment(l, m, i, d)
            alignments[case_index, beam_index, :, :a.shape[1]] = a
    return alignments
