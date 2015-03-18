import rlpy.Tools.results as rt

paths = { "LSPI": "./Results/Tests/gridworld/LSPI",
	"PSRL" : "./Results/Tests/gridworld/PSRL",
     "UCRL" : "./Results/Tests/gridworld/UCRL",
     "SARSA" : "./Results/Tests/gridworld/SARSA"}

merger = rt.MultiExperimentResults(paths)
fig = merger.plot_avg_sem("learning_steps","return")
rt.save_figure(fig,"./Results/Tests/gridworld/plot.pdf")
