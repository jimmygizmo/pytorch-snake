# plotter.py  -  Importable module for usage by the ML training agent.
# Simple plot of ongoing training progress.

import matplotlib.pyplot as plt
from IPython import display


# ###################################    TYPE DEFINITIONS, GLOBAL INITIALIZATION    ####################################

plt.ion()


# #############################################    FUNCTION DEFINITIONS    #############################################

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('ML Model trained by each game of Snake')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)


##
#


# IPython  (Backend for Jupyter notebook and related to plotting and plots seen in IDEs etc.)
# TODO: Clarify/improve the definition of IPython
# https://ipython.org/

# IPYthon Tutorial:
# https://ipython.readthedocs.io/en/stable/interactive/tutorial.html


##
#
