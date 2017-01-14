import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D



def draw(id=0,label=None,data=None,major=None,other=None,whole_data=None):
    fig = plt.figure(id, figsize=(12, 8))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    plt.cla()
    if isinstance(label,pd.DataFrame):
        y = label.as_matrix().astype(np.float)
    if isinstance(label,list):
        y = np.array(label).astype(np.float)
    if isinstance(label,np.ndarray):
        y = label.astype(np.float)
    z = np.array([0.0 for _ in range(len(data))])
    ax.scatter(data[:, 0], data[:, 1], data[:,2], c=y)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if major is not None and other is not None:
        d = whole_data[major[0]]
        ax.text(d[0],d[1],0.0,major[1])
        for _ in range(len(other)):
            d = data[other[_][0]]
            t = other[_][1]
            ax.text(d[0],d[1],0.0,t)