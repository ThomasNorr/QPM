import numpy as np
import torch

from evaluation.Metrics.Dependence import compute_contribution_top_feature
from evaluation.Metrics.cub_Alignment import get_cub_alignment_from_features
from evaluation.diversity import MultiKCrossChannelMaxPooledSum
from evaluation.utils import get_metrics_for_model


def evaluateALLMetricsForComps(features_train,  outputs_train,  feature_maps_test,
                               outputs_test, linear_matrix,  labels_train):
    with torch.no_grad():
        if len(features_train) < 7000: # recognize CUB and TravelingBirds
            cub_alignment = get_cub_alignment_from_features(features_train)
        else:
            cub_alignment = 0
        print("cub_alignment: ", cub_alignment)
        localizer = MultiKCrossChannelMaxPooledSum(range(1, 6), linear_matrix, None)
        batch_size = 300
        for i in range(np.floor(len(features_train) / batch_size).astype(int)):
            localizer(outputs_test[i * batch_size:(i + 1) * batch_size].to("cuda"),
                      feature_maps_test[i * batch_size:(i + 1) * batch_size].to("cuda"))

        locality, exlusive_locality = localizer.get_result()
        diversity = locality[4]
        print("diversity@5: ", diversity)
        abs_frac_mean = compute_contribution_top_feature(
            features_train,
            outputs_train,
            linear_matrix,
     labels_train)
        print("Dependence ", abs_frac_mean)
        answer_dict = {"diversity": diversity.item(),  "Dependence": abs_frac_mean.item(), "Alignment":cub_alignment}
    return answer_dict

def eval_model_on_all_qsenn_metrics(model, test_loader, train_loader):
    return get_metrics_for_model(train_loader, test_loader, model, evaluateALLMetricsForComps)


