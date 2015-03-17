from rlpy.Tools.run import run 

run("examples/mdp_chain/mdp_chain_post.py","./Results/Tests/mdp_chain/PSRL",ids=range(5), parallelization ="joblib")

run("examples/mdp_chain/mdp_chain_lspi.py","./Results/Tests/mdp_chain/LSPI",ids=range(5), parallelization ="joblib")

run("examples/mdp_chain/mdp_chain_sarsa.py","./Results/Tests/mdp_chain/SARSA",ids=range(5), parallelization ="joblib")

run("examples/mdp_chain/mdp_chain_ucrl.py","./Results/Tests/mdp_chain/UCRL",ids=range(5), parallelization ="joblib")
