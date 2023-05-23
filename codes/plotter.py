import matplotlib.pyplot as plt
import numpy as np
def scatter1d(tf):
    plt.plot(tf,np.zeros_like(tf))
    plt.show()
def scatter2d(tf):
    for i in tf:
        plt.scatter(i[0],i[1])
    plt.show()
def scatter3d(tf):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    for i in tf:
        ax.scatter(i[0],i[1],i[2])
    plt.show()