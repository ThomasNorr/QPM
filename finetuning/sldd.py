import numpy as np
import torch

from FeatureDiversityLoss import FeatureDiversityLoss
from finetuning.utils import train_n_epochs
from sparsification.glmBasedSparsification import compute_feature_selection_and_assignment
from sparsification.sldd import compute_sldd_feature_selection_and_assignment
from train import train, test
from training.optim import get_optimizer




def finetune_sldd(model, train_loader, test_loader, log_dir, n_classes, seed, beta, optimization_schedule,n_per_class, n_features, ):
        feature_sel, weight, bias, mean, std = compute_sldd_feature_selection_and_assignment(model, train_loader,
                                                                                        test_loader,
                                                                                        log_dir, n_classes, seed,n_per_class, n_features)
        model.set_model_sldd(feature_sel, weight, mean, std, bias)
        model = train_n_epochs( model, beta, optimization_schedule, train_loader, test_loader)
        return model


