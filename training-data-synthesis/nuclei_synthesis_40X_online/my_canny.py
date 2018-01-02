import numpy as np
from skimage import feature

def canny_edge(im, sig=3, low_th=12, high_th=80):
    imgray = np.mean(im, axis = 2);
    edges = feature.canny(imgray, sigma=sig, low_threshold=low_th, high_threshold=high_th);
    return edges;

def canny_edge_on_mask(imgray):
    edges = feature.canny(imgray, sigma=1.5, low_threshold=12, high_threshold=80);
    edges_ret = edges.copy();
    if np.random.rand() < 0.66:
        edges_ret[:, :-1] += edges[:, 1:];
    if np.random.rand() < 0.66:
        edges_ret[:, 1:] += edges[:, :-1];
    if np.random.rand() < 0.66:
        edges_ret[:-1, :] += edges[1:, :];
    if np.random.rand() < 0.66:
        edges_ret[1:, :] += edges[:-1, :];
    return (edges_ret>0).astype(np.uint8);

