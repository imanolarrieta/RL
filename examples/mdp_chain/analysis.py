import rlpy.Tools.results as rt

paths = { "LSPI": "./Results/Tests/mdp_chain/LSPI",
	"PSRL" : "./Results/Tests/mdp_chain/PSRL",
     "UCRL" : "./Results/Tests/mdp_chain/UCRL",
     "SARSA" : "./Results/Tests/mdp_chain/SARSA"}

merger = rt.MultiExperimentResults(paths)
fig = merger.plot_avg_sem("learning_steps","return")
rt.save_figure(fig,"./Results/Tests/plot.pdf")
