#******************************************************************************
#
#   Keras Gems
#   Copyright 2018 Moritz Becher, Lukas Prantl and Steffen Wiewel
#
#   progress bar console application
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

# show iterations progress in progress bar
def progress_bar(iteration, total, prefix = '', suffix = '', decimals = 0, length = 100, fill = '='):
    """
    Call in a loop to create terminal progress bar 
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * (filledLength) + '>' + '-' * (length - filledLength - 1)
    if iteration == total:
        bar = fill * (filledLength)
    print("\r{} [{}] {}% {}".format(prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()