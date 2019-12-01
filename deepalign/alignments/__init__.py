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

import inspect
import sys

from deepalign.alignments.confnet import ConfNet
from deepalign.alignments.baseline import *

# Lookup dict for AD abbreviations
ALIGNERS = dict((ad.abbreviation, ad) for _, ad in inspect.getmembers(sys.modules[__name__], inspect.isclass)
                if hasattr(ad, 'abbreviation') and ad.abbreviation is not None)