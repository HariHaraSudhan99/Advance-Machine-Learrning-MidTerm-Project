import numpy as np
from sklearn.decomposition import NMF
from sklearn.neighbors import kneighbors_graph
from utils import buildGinv
import warnings

warnings.filterwarnings('ignore')

def run_GPCA(adata, location=None, n_components=50, _lambda=0.5, save_reconstruction=False):

    Expr = adata.X
    Ginv = buildGinv(adata, location, _lambda)
    
    C = np.dot(np.dot(Expr.T, Ginv), Expr)
    _ , W = np.linalg.eigh(C)
    W = W[:, ::-1]
    W = W[:, :n_components]
    Z = np.dot(np.dot(Ginv, Expr), W)

    if save_reconstruction:
        adata.layers["GraphPCA_ReX"] = np.dot(Z, W.T)

    return Z, W

def run_GNMF(adata, location=None, n_components=50, _lambda=0.5, save_reconstruction=False, **nmf_kwargs):

    Expr = adata.X
    Ginv = buildGinv(adata, location, _lambda)

    nmf = NMF(n_components=n_components, random_state=42, **nmf_kwargs)
    Z = nmf.fit_transform(np.dot(Ginv, Expr))
    W = nmf.components_.T
    
    if save_reconstruction:
        adata.layers["GraphNMF_ReX"] = np.dot(Z, W.T)

    return Z, W
