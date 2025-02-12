import numpy as np
import sklearn
import torch
from tqdm import trange

from evaluation.Metrics.Contrastiveness import gmm_overlap_per_feature
from evaluation.Metrics.Correlation import get_correlation
from evaluation.Metrics.Dependence import compute_contribution_top_feature, compute_class_independence
from evaluation.Metrics.StructuralGrounding import get_structural_grounding_for_weight_matrix
from evaluation.diversity import MultiKCrossChannelMaxPooledSum
from evaluation.utils import get_metrics_for_model


def evaluateALLMetricsForComps(features_train,  outputs_train,  feature_maps_test,
                               outputs_test, linear_matrix,  labels_train):
    # Calculate Diversity, Dependency, GMM Overlap and similarity with CUB GT for given features

    with torch.no_grad():
        if len(features_train) < 7000:
            cub_overlap = get_structural_grounding_for_weight_matrix(linear_matrix)
        else:
            cub_overlap = 0
        print("cub_overlap: ", cub_overlap)
        soft_max_scaled_localizer = MultiKCrossChannelMaxPooledSum(range(1, 6), linear_matrix, None,
                                                                   func="SumNMax")
        batch_size = 300
        for i in range(np.floor(len(features_train) / batch_size).astype(int)):
            soft_max_scaled_localizer(outputs_test[i * batch_size:(i + 1) * batch_size].to("cuda"),
                                      feature_maps_test[i * batch_size:(i + 1) * batch_size].to("cuda"))
        diversity_sm_scaled = soft_max_scaled_localizer.get_result()[0][4].item()
        print("SID@5: ", diversity_sm_scaled)
        if features_train.shape[1] > 1000:
            print("Skipping Contrastiveness for dense model as it takes a while")
            overlap_mean = -1
        else:
            overlap_mean = 1 - gmm_overlap_per_feature(features_train).mean()
        class_independence = compute_class_independence(features_train,  linear_matrix,
                                                                     labels_train)
        correlation_features = get_correlation(features_train).item()
        answer_dict = {"SID@5": diversity_sm_scaled,"Class-Independence": class_independence, "Contrastiveness": overlap_mean,"Structural Grounding": cub_overlap, "Correlation":correlation_features}
    return answer_dict

def eval_model_on_all_qpm_metrics(model, test_loader, train_loader):
    # TODO evaluate model on all metrics
    # TODO return metrics
    return get_metrics_for_model(train_loader, test_loader, model, evaluateALLMetricsForComps)
