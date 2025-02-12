import torch


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt
def compute_cross_corr(features_train):
    features_train = torch.tensor(features_train)
    cross_correlation_matrix = sim_matrix(torch.transpose(features_train, 1, 0),
                                          torch.transpose(features_train, 1, 0))
    return cross_correlation_matrix


def get_correlation(features):
    cross_corr_matrix = compute_cross_corr(features)
    cross_corr_matrix[torch.eye(cross_corr_matrix.shape[0], dtype=torch.bool)] = 0
    max_per_feature = cross_corr_matrix.max(dim=0)[0]
    return max_per_feature.mean()