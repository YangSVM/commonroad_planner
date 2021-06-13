import numpy as np
import matplotlib.pyplot as plt



def plot_lanelet_network(lanelet_network):
    for lanelet in lanelet_network.lanelets:
        plot_lanelet(lanelet)
    plt.show()

def plot_lanelet(lanelet):
    lv = lanelet.left_vertices
    rv = lanelet.right_vertices
    plt.plot(lv[:,0], lv[:,1], 'k-')
    plt.plot(rv[:,0], rv[:,1],  'k-')
    start_line = np.array([lv[0,:], rv[0,:]])
    end_line = np.array([lv[-1,:], rv[-1,:]])
    plt.plot(start_line[:,0], start_line[:,1], 'r--')
    plt.plot(end_line[:,0], end_line[:,1], 'r--')
    
    mid_id = int((len(lv))/2)
    center = (lv[mid_id, :] + rv[mid_id, :])/2
    arror_end_id = min(mid_id+1, len(lv))
    arror_end = (lv[arror_end_id, :] + rv[arror_end_id, :])/2
    plt.plot(center[0], center[1], 'r*')
    # plt.arrow(center[0], center[1], arror_end[0] - center[0], arror_end[1]- center[1])
    plt.annotate(str(lanelet.lanelet_id), (center[0], center[1]) )
    plt.annotate('', xytext=(center[0], center[1]), xy=(arror_end[0], arror_end[1]), arrowprops=dict(arrowstyle="->"))
    # plt.show()
