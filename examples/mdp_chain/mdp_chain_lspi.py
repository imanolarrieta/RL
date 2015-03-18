
"""
Agent Tutorial for RLPy
=================================

Assumes you have created the SARSA0.py agent according to the tutorial and
placed it in the current working directory.
Tests the agent on the GridWorld domain.
"""
__author__ = "Robert H. Klein"
from rlpy.Domains import ChainMDP
from rlpy.Agents import LSPI
from rlpy.Representations import Tabular
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import os


def make_experiment(exp_id=1, path="./Results/Tests/mdp_chain-lspi"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """
   # Experiment variables
    opt = {}
    opt["path"] = path
    opt["exp_id"] = exp_id
    opt["max_steps"] = 30000
    opt["num_policy_checks"] = 10
    opt["checks_per_policy"] = 50

    ## Domain:
    chain_size = 10
    domain = ChainMDP(chain_size)
    opt["domain"] = domain

    ## Representation
    # discretization only needed for continuous state spaces, discarded otherwise
    representation  = Tabular(domain, discretization=20)

     # Policy
    policy = eGreedy(representation, epsilon=0.2)

    # Agent
    opt["agent"] = LSPI(policy, representation, domain.discount_factor,
                 opt["max_steps"], 1000)
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    experiment = make_experiment(1)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=True,  # show policy / value function?
                   visualize_performance=1)  # show performance runs?
    experiment.plot()
    experiment.save()
