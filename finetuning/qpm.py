from finetuning.utils import train_n_epochs
from sparsification.qpm_sparsification import compute_qpm_feature_selection_and_assignment


def finetune_qpm(model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule, n_features, n_per_class):
    feature_sel, weight,  mean, std = compute_qpm_feature_selection_and_assignment(model, train_loader,
                                                                                         test_loader,
                                                                                         log_dir, n_classes, seed, n_features, n_per_class)
    model.set_model_sldd(feature_sel, weight, mean, std, retrain_normalisation=False)
    model = train_n_epochs(model, beta, optimization_schedule, train_loader, test_loader)
    return model