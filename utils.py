import random
import numpy as np
import pandas as pd
import scipy
import scanpy as sc
from sklearn.cluster import AgglomerativeClustering, BisectingKMeans, KMeans, SpectralClustering, MiniBatchKMeans, FeatureAgglomeration
from sklearn.metrics import pairwise_distances as pair
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.metrics import calinski_harabasz_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph
from scipy.sparse import issparse
import warnings

warnings.filterwarnings('ignore')

def set_random_state(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def buildGinv(adata, location=None, _lambda=0.5):

    global graphL
    Expr = adata.X
    n_neighbors = 6
    
    if issparse(Expr):
        Expr = Expr.todense()

    graph = kneighbors_graph(np.asarray(location), int(n_neighbors), metric='euclidean',
                                 metric_params={}, include_self=False)
    graph = 0.5 * (graph + graph.T)
    graphL = csgraph.laplacian(graph, normed=False)

    G = scipy.sparse.eye(adata.shape[0]) + _lambda * graphL
    if issparse(G):
        Ginv = np.array(np.linalg.inv(G.todense()))
    else:
        Ginv = np.array(np.linalg.inv(G))

    return Ginv

def preprocessGPCA(adata):

    sc.pp.scale(adata)
    x_array = adata.obs.array_row.tolist()
    y_array = adata.obs.array_col.tolist()
    location = np.array([x_array, y_array]).T.astype(np.float32)

    return adata, location

def preprocessGNMF(adata):

    sc.pp.scale(adata)
    X_new = adata.T.X - adata.T.X.min(axis=1).reshape(-1, 1)
    adata.X = X_new.T
    x_array = adata.obs.array_row.tolist()
    y_array = adata.obs.array_col.tolist()
    location = np.array([x_array, y_array]).T.astype(np.float32)

    return adata, location

def clusterAlgorithm(Z, n_clusters, algorithm):

    res = None

    if algorithm == "KMeans":        
        estimator = KMeans(n_clusters=n_clusters)
        res = estimator.fit(Z)
    elif algorithm == "BisectingKMeans":
        estimator = BisectingKMeans(n_clusters=n_clusters)
        res = estimator.fit(Z)
    elif algorithm == "MiniBatchKMeans":
        estimator = MiniBatchKMeans(n_clusters=n_clusters)
        res = estimator.fit(Z)
    elif algorithm == "AgglomerativeClustering":
        estimator = AgglomerativeClustering(n_clusters=n_clusters)
        res = estimator.fit(Z)
    elif algorithm == "FeatureAgglomeration":
        estimator = FeatureAgglomeration(n_clusters=n_clusters)
        res = estimator.fit(Z)
    elif algorithm == "SpectralClustering":
        estimator = SpectralClustering(n_clusters=n_clusters)
        res = estimator.fit(Z)
    else:
        print("Using KMeans since no proper algorithms where mentioned")
        estimator = KMeans(n_clusters=n_clusters)
        res = estimator.fit(Z)
    
    return res.labels_

def bestPerforming(adata, Z, n_clusters, algorithm):
    
    kMeansParams = {
                        "init" : ["k-means++", "random"],
                        "n_init" : [1,2,5,10,20],
                        "tol" : [1e-3, 1e-4, 1e-5],
                        "algorithm" : ["lloyd", "elkan"]
                   }
    biKMeansParams = {
                        "init" : ["k-means++", "random"],
                        "n_init" : [1,2,5,10,20],
                        "tol" : [1e-3, 1e-4, 1e-5],
                        "algorithm" : ["lloyd", "elkan"],
                        "bisecting_strategy" : ["biggest_inertia", "largest_cluster"]
                     }
    mbKMeansParams = {
                        "init" : ["k-means++", "random"],
                        "n_init" : [1,2,5,10,20],
                        "tol" : [1e-3, 1e-4, 1e-5],
                        "batch_size" : [64, 128, 256,512, 1024, 2046]
                     }
    aggParams = {
                    "linkage" : ["ward", "complete", "average", "single"]
                }
    featAggParams = {
                        "linkage" : ["ward", "complete", "average", "single"],
                        "pooling_func" : [np.mean, np.min, np.max]
                    }
    specParams = {
                    "eigen_solver" : ["arpack", "lobpcg", "amg"],
                    "n_init" : [1,2,5,10,20],
                    "gamma" : [0.1, 0.5, 1.0, 1.5 ,2.0],
                    "affinity" : ["rbf", "nearest_neighbors"],
                    "n_neighbors" : [2,3,5,7,9,10,12,15],
                    "assign_labels" : ["kmeans", "discretize", "cluster_qr"]
                 }

    kMeansCV = GridSearchCV(KMeans(n_clusters=n_clusters), kMeansParams, cv = 2, n_jobs = -1)
    kMeansCV.fit(Z)
    kMeans = kMeansCV.best_estimator_
    kMeans_labels = kMeans.labels_

    bikMeansCV = GridSearchCV(BisectingKMeans(n_clusters=n_clusters), biKMeansParams, scoring=make_scorer(ari_score), cv = 2, n_jobs = -1)
    bikMeansCV.fit(Z)
    bikMeans = bikMeansCV.best_estimator_
    bikMeans_labels = bikMeans.labels_

    mbkMeansCV = GridSearchCV(MiniBatchKMeans(n_clusters=n_clusters), mbKMeansParams, scoring=make_scorer(ari_score), cv = 2, n_jobs = -1)
    mbkMeansCV.fit(Z)
    mbkMeans = mbkMeansCV.best_estimator_
    mbkMeans_labels = mbkMeans.labels_

    aggCV = GridSearchCV(AgglomerativeClustering(n_clusters=n_clusters), aggParams, scoring=make_scorer(ari_score), cv = 2, n_jobs = -1)
    aggCV.fit(Z)
    agg = aggCV.best_estimator_
    agg_labels = agg.labels_

    featAggCV = GridSearchCV(FeatureAgglomeration(n_clusters=n_clusters), featAggParams, scoring=make_scorer(ari_score), cv = 2, n_jobs = -1)
    featAggCV.fit(Z)
    featAgg = featAggCV.best_estimator_    
    featAgg_labels = featAgg.labels_

    specCV = GridSearchCV(SpectralClustering(n_clusters=n_clusters), specParams, scoring=make_scorer(ari_score), cv = 2, n_jobs = -1)
    specCV.fit(Z)
    spec = specCV.best_estimator_
    spec_labels = spec.labels_

    print("ARI score of KMeans is",)
    print("ARI score of KMeans is",)
    print("ARI score of KMeans is",)
    print("ARI score of KMeans is",)
    print("ARI score of KMeans is",)
    print("ARI score of KMeans is",)

    pass

def plot(adata, location, lable_pred, Z, method):

    assert method in ["GPCA_pred", "GNMF_pred"], "Invalid method . Must be 'GPCA_pred' or 'GNMF_pred'."

    adata.obs[method] = lable_pred
    adata.obs[method] = adata.obs[method].astype('category')
    refined_pred=refine(sample_id=adata.obs.index.tolist(), 
            pred=adata.obs[method].tolist(), dis= pair(location))
    adata.obs[method] = refined_pred
    adata.obs[method] = adata.obs[method].astype('category')

    print("ARI = {}".format(ari_score(adata.obs.ground_truth, adata.obs[method])))
    print("NMI = {}".format(normalized_mutual_info_score(adata.obs.ground_truth, adata.obs[method])))
    print("Silhouette = {}".format(silhouette_score(Z, adata.obs[method].cat.codes)))
    print("CH = {}".format(calinski_harabasz_score(Z, adata.obs[method])))

    spatial_coords = adata.obsm["spatial"]
    adata.obs["x_coord"] = spatial_coords[:, 0]
    adata.obs["y_coord"] = spatial_coords[:, 1]

    sc.pl.scatter(adata, x="x_coord", y="y_coord", color=["ground_truth", method])

def refine(sample_id, pred, dis):
    
    global num_nbs
    refined_pred = []
    pred = pd.DataFrame({"pred": pred}, index=sample_id)
    dis_df = pd.DataFrame(dis, index=sample_id, columns=sample_id)
    num_nbs = 6
        
    for i in range(len(sample_id)):
        index = sample_id[i]
        dis_tmp = dis_df.loc[index, :].sort_values()
        nbs = dis_tmp[0:num_nbs + 1]
        nbs_pred = pred.loc[nbs.index, "pred"]
        self_pred = pred.loc[index, "pred"]
        v_c = nbs_pred.value_counts()
        
        if (v_c.loc[self_pred] < num_nbs / 2) and (np.max(v_c) > num_nbs / 2):
            refined_pred.append(v_c.idxmax())
        else:
            refined_pred.append(self_pred)
    
    return refined_pred
