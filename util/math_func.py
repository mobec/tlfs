#******************************************************************************
#
#   Keras Gems
#   Copyright 2018 Moritz Becher, Lukas Prantl and Steffen Wiewel
#
#   required math functions
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
# 
#       http://www.apache.org/licenses/LICENSE-2.0
# 
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
#******************************************************************************

from math import log10, floor

# https://stackoverflow.com/a/3413529
def round_sig(x, sig=2):
    if x == 0.0:
        return round(x, sig)
    return round(x, sig-int(floor(log10(abs(x))))-1)

# https://stackoverflow.com/a/3928583
def round_tuple(tup, sig=2):
    if isinstance(tup, tuple):
        return tuple(map(lambda x: isinstance(x, float) and round_sig(x, sig) or x, tup))
    else:
        return tup