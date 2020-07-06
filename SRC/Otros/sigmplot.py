# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 19:48:44 2019

@author: Mauricio
"""
import os

# Dependency imports
import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.backends import backend_agg
import numpy as np


fig = figure.Figure(figsize=(6, 6))
canvas = backend_agg.FigureCanvasAgg(fig)
ax = fig.add_subplot(1, 1, 1)

def sigmoid(x):
    sigm = 1. / (1. + np.exp(-x))
    return sigm
 
def plot_sigm(w, b, **kwargs):
    x1s = np.linspace(-7, 7, 100)
    x1ss = w*x1s + b
    x2s = sigmoid(x1ss)
    ax.plot(x1s, x2s, **kwargs)
  
w, b = 1, 0
plot_sigm(w, b, lw=2, color="blue", label = 'p = Sigm(x)')
#ax.plot(np.linspace(-7, 7, 100), np.ones(100),lw=1, color="black")     
 
ax.set_xlim([-7, 7])
ax.set_ylim([-0.1, 1.2])

ax.set_title('Función Sigmoide')
ax.set_xticks(np.arange(-7,8,1))
ax.set_yticks(np.arange(-0.1,1.3,0.1))
ax.xaxis.grid(b=True)
ax.yaxis.grid(b=True)
ax.spines['left'].set_position(('data', 0))
ax.spines['right'].set_color('none')
ax.spines['bottom'].set_position(('data', 0.0))
ax.spines['top'].set_position(('data', 1.0))
#ax.spines['top'].set_color('none')
ax.set_xlabel('x', labelpad=15, fontsize=14)
ax.set_ylabel('p', labelpad = 150, fontsize=14)



ax.legend()
fname=os.path.join(r"SRC/Otros",
                                         "Sigmoide_"+str(w)+"_"+str(b)+".png")

canvas.print_figure(fname, format="png")
print("saved {}".format(fname))