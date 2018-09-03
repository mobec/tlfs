#******************************************************************************
#
#   Keras Gems
#   Copyright 2018 Moritz Becher, Lukas Prantl and Steffen Wiewel
#
#   hyper search class
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

from ..models.architecture import Network
import json
from itertools import product

#   Initialization
#   - instantiate a neural network that is a subclass of 'Network'
#   - create team of hyperparameters as list of HyperParameter objects
#       e.g. [HyperParameter("learning_rate", [0.0001, 0.1], 20), HyperParameter("regularization", [0.1, 0.2], 5)]

#   Usage
#   - call 'search' with the number of epochs each training run should execute
#   - in this function all permutations of the hyper team get evaluated and executed
#   - the history with corresponding hyper values is stored for each run

class HyperSearch(object):
    """ Superclass for all hyperparameter searches """
    #---------------------------------------------------------------------------------
    def __init__(self, network, param_team, output_folder=None, **kwargs):
        assert isinstance(network, Network), "Variable 'network' must be instance of 'Network' class"
        self.network = network
        assert isinstance(param_team, list), "Variable 'param_team' must be instance of 'list' class"
        self.param_team = param_team
        self.output_folder = output_folder
        ## Subclass initialization
        #self._initialize(**kwargs)

    #---------------------------------------------------------------------------------
    def search(self, epochs, **kwargs):
        """ Trains the network for all configurations and searches best history """
        # search_hist dictionary maps values for parameters to resulting training history
        # -> { (param0, param1, param2, ...): (history) }
        search_hist = {}

        # returns tuples of all valid hyperparameter configurations
        # e.g. [(1, 'a', 4), (1, 'a', 5), (1, 'b', 4)]
        permutations = self._build_team_permutations(**kwargs)

        # iterate all permutations of form perm_it=(param0, param1, param2, ...)
        for perm_it in permutations:
            # build dictionary of current hyperparameter values for training
            # e.g. current_params = {"learning_rate": 0.1, "regularizer": 0.02, ...}
            current_params = {}
            for param_idx, param in enumerate(self.param_team):
                current_params[param.name] = perm_it[param_idx] # get current value by index

            self.network.update_parameters(current_params)
            current_hist = self.network.train(epochs, **kwargs)
            # add current_hist to complete search history in form { (param0, param1, param2, ...): (history) }
            search_hist[perm_it] = current_hist.history

            # write out current histories (so nothing gets lost if something goes wrong)
            if search_hist is not None:
                with open(self.output_folder + "search_hist.json", 'w') as f:
                    json.dump({str(k):v for k, v in search_hist.items()}, f, indent=4)

        return search_hist

    #---------------------------------------------------------------------------------
    def _build_team_permutations(self, **kwargs):
        """ All permutations of the hyper team are evaluated and returned as a list of tuples """
        # use itertools.product to generate said tuples
        # https://docs.python.org/3/library/itertools.html#itertools.product

        # [HyperParameter("learning_rate", [0.0001, 0.1], 20), HyperParameter("regularization", [0.1, 0.2], 5)]
        
        # build up list of parameter values
        # iterables = [ [param0_0, param0_1, param0_2], [param1_0, param1_1] ]
        iterables = []
        for param in self.param_team:
            iterables.append(param.get_values())
        # build the product of previously created list that holds all permutations
        return product(*iterables)

    #---------------------------------------------------------------------------------