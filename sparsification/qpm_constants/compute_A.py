import numpy as np
import torch
from tqdm import trange


def corr_matrix(features, labels):
    # features: (n_samples, n_features)
    # labels: (n_samples)
    n_samples, n_features = features.shape
    n_classes = labels.max() + 1
    corr_matrix = np.zeros((n_features, n_classes))

    for class_idx in trange(n_classes):
        class_labels = labels == class_idx
        mean_less_labels = class_labels - class_labels.mean()
        mean_less_features = features - features.mean(axis=0)
        corr_matrix[:, class_idx] = (mean_less_features.T @ mean_less_labels) / mean_less_labels.shape[0] / (
                mean_less_features.std(axis=0) * mean_less_labels.std())
    return corr_matrix


def compute_feat_class_corr_matrix(train_loader):
    features, labels = [], []
    for data in train_loader:
        features.append(data[0])
        labels.append(data[1])
    features = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()
    return corr_matrix(features, labels)