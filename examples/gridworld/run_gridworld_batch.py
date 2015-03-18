from rlpy.Tools.run import run 

run("examples/gridworld/posterior_sampling.py","./Results/Tests/gridworld/PSRL",ids=range(5), parallelization ="joblib")

run("examples/gridworld/lspi.py","./Results/Tests/gridworld/LSPI",ids=range(5), parallelization ="joblib")

run("examples/gridworld/sarsa.py","./Results/Tests/gridworld/SARSA",ids=range(5), parallelization ="joblib")

run("examples/gridworld/ucrl.py","./Results/Tests/gridworld/UCRL",ids=range(5), parallelization ="joblib")
