from statistics import NormalDist

import numpy as np
from sklearn.mixture import GaussianMixture
from tqdm import trange


def gmm_overlap_per_feature(features):
    n_features = features.shape[1]
    answer = np.zeros(n_features)
    for i in trange(n_features):
            answer[i] = gmm_overlap(features[:, i])
    return answer

def gmm_overlap(feature):
    gmm = GaussianMixture(n_components=2)
    gmm.fit(feature.reshape(-1, 1))
    overlap = get_overlap(gmm)
    return overlap

def get_overlap(mixture_model):
    return NormalDist(mu=mixture_model.means_[0], sigma=np.sqrt(mixture_model.covariances_[0])).overlap(
        NormalDist(mu=mixture_model.means_[1], sigma=np.sqrt(mixture_model.covariances_[1])))
