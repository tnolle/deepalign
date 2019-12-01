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

import numpy as np


class Penalties:
    MATCH = 1
    MISMATCH = -np.inf
    GAP = -1


PADDING_VAL = 0


def align(a, b):
    a = np.trim_zeros(a)
    m = len(a) + 1

    beam = np.trim_zeros(b)
    n = len(beam) + 1  # Technically, all beams should have the same length, but better safe then sorry :)

    # Initialize alignment and direction matrices
    alignment_matrix = np.zeros((m, n))
    direction_matrix = np.zeros((m, n, 3))  # 0 = Diagonal, 1 = Vertical, 2 = Horizontal

    # Fill first row with gap penalty
    alignment_matrix[:, 0] = np.arange(m) * Penalties.GAP
    direction_matrix[:, 0, :] = np.repeat([[0, 0, 1]], m, axis=0)
    direction_matrix[0, :, :] = np.repeat([[0, 1, 0]], n, axis=0)

    # Fill first column with gap penalty
    alignment_matrix[0, :] = np.arange(n) * Penalties.GAP
    direction_matrix[0, 0, :] = [1, 0, 0]

    # Build matrices
    for i in range(1, m):
        for j in range(1, n):
            penalty = Penalties.MATCH if a[i - 1] == beam[j - 1] else Penalties.MISMATCH

            d = alignment_matrix[i - 1][j - 1] + penalty  # Diagonal / Top-Left - Match / Mismatch
            v = alignment_matrix[i - 1][j] + Penalties.GAP  # Vertical / Top - Gap
            h = alignment_matrix[i][j - 1] + Penalties.GAP  # Horizontal / Left - Gap

            candidates = np.array([d, v, h])
            max_val = np.amax(candidates)

            alignment_matrix[i][j] = max_val
            direction_matrix[i][j] = (candidates == max_val) * 1

    # Init stack with lower-right corner and empty alignments
    stack = [(m - 1, n - 1, [], [])]
    alignments = []
    if len(stack) > 0:
        i, j, alignment_a, alignment_b = stack.pop()

        # Trace back path
        while i > 0 or j > 0:
            c_dir = direction_matrix[i][j]

            old_i, old_j = i, j
            already_moved = False

            # Diagonal - Match / Mismatch
            if c_dir[0] > 0:
                alignment_a.append(a[i - 1])
                alignment_b.append(beam[j - 1])
                i -= 1
                j -= 1
                already_moved = True

            # Vertical - Gap in Beam
            if c_dir[1] > 0:
                if not already_moved:
                    alignment_a.append(a[i - 1])
                    alignment_b.append(PADDING_VAL)
                    i -= 1
                    already_moved = True
                else:
                    alignment_a_ = alignment_a[:-1].copy()
                    alignment_b_ = alignment_b[:-1].copy()
                    alignment_a_.append(a[old_i - 1])
                    alignment_b_.append(PADDING_VAL)
                    stack.append((old_i - 1, old_j, alignment_a_, alignment_b_))

            # Horizontal - Gap in Original
            if c_dir[2] > 0:
                if not already_moved:
                    alignment_a.append(PADDING_VAL)
                    alignment_b.append(beam[j - 1])
                    j -= 1
                    already_moved = True
                else:
                    alignment_a_ = alignment_a[:-1].copy()
                    alignment_b_ = alignment_b[:-1].copy()
                    alignment_a_.append(PADDING_VAL)
                    alignment_b_.append(beam[old_j - 1])
                    stack.append((old_i, old_j - 1, alignment_a_, alignment_b_))

        alignment_a.reverse()
        alignment_b.reverse()

        alignments.append(np.array([alignment_a, alignment_b]))

    return alignments


def needleman_wunsch(a, b):
    if isinstance(a[0], list) and isinstance(b[0], list):
        return [align(_a, _b) for _a, _b in zip(a, b)]
    else:
        return align(a, b)
