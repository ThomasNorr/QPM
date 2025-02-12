import logging
import os
import shutil

import numpy as np
import pandas as pd
import torch
from glm_saga.elasticnet import glm_saga
from torch import nn

from sparsification.FeatureSelection import FeatureSelectionFitting
from sparsification import data_helpers
from sparsification.utils import get_default_args, compute_features_and_metadata, select_in_loader, get_feature_loaders


def get_glm_selection(feature_loaders, metadata,  args, num_classes, device, n_features_to_select, folder):
    num_features = metadata["X"]["num_features"][0]
    fittingClass = FeatureSelectionFitting(num_features, num_classes, args, 0.8,
                                           n_features_to_select,
                                           0.1,folder,
                                           lookback=3, tol=1e-4,
                                           epsilon=1,)
    to_drop, test_acc = fittingClass.fit(feature_loaders, metadata, device)
    selected_features = torch.tensor([i for i in range(num_features) if i not in to_drop])
    return selected_features


def compute_feature_selection_and_assignment(model, train_loader, test_loader, log_folder,num_classes, seed,  select_features = 50):
    feature_loaders, metadata, device,args =  get_feature_loaders(seed, log_folder,train_loader, test_loader, model, num_classes, )

    if os.path.exists(log_folder / f"SlDD_Selection_{select_features}.pt"):
        feature_selection = torch.load(log_folder / f"SlDD_Selection_{select_features}.pt")
    else:
        used_features = model.linear.weight.shape[1]
        if used_features != select_features:
            selection_folder = log_folder / "sldd_selection" # overwrite with None to prevent saving
            feature_selection = get_glm_selection(feature_loaders, metadata,   args,
                                      num_classes,
                                      device,select_features, selection_folder
                                      )
        else:
            feature_selection = model.linear.selection
        torch.save(feature_selection, log_folder / f"SlDD_Selection_{select_features}.pt")
    feature_loaders = select_in_loader(feature_loaders, feature_selection)
    mean, std = metadata["X"]["mean"], metadata["X"]["std"]
    mean_to_pass_in = mean
    std_to_pass_in = std
    if len(mean) != feature_selection.shape[0]:
        mean_to_pass_in = mean[feature_selection]
        std_to_pass_in = std[feature_selection]

    sparse_matrices, biases = fit_glm(log_folder, mean_to_pass_in, std_to_pass_in, feature_loaders, num_classes, select_features)

    return feature_selection, sparse_matrices, biases, mean, std


def fit_glm(log_dir,mean, std , feature_loaders,  num_classes,   select_features = 50):
    output_folder = log_dir / "glm_path"
    if not output_folder.exists() or len(list(output_folder.iterdir())) != 102:
        shutil.rmtree(output_folder, ignore_errors=True)
        output_folder.mkdir(exist_ok=True, parents=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        linear = nn.Linear(select_features, num_classes).to(device)
        for p in [linear.weight, linear.bias]:
            p.data.zero_()
        print("Preparing normalization preprocess and indexed dataloader")
        metadata = {"X": {"mean": mean, "std": std},}
        preprocess = data_helpers.NormalizedRepresentation(feature_loaders['train'],
                                                           metadata=metadata,
                                                           device=linear.weight.device)

        print("Calculating the regularization path")
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
        params = glm_saga(linear,
                          feature_loaders['train'],
                          0.1,
                          2000,
                          0.99, k=100,
                          val_loader=feature_loaders['val'],
                          test_loader=feature_loaders['test'],
                          n_classes=num_classes,
                          checkpoint=str(output_folder),
                          verbose=200,
                          tol=1e-4, # Change for ImageNet
                          lookbehind=5,
                          lr_decay_factor=1,
                          group=False,
                          epsilon=0.001,
                          metadata=None, # To let it be recomputed
                          preprocess=preprocess, )
    results = load_glm(output_folder)
    sparse_matrices = results["weights"]
    biases =  results["biases"]

    return sparse_matrices, biases

def load_glm(result_dir):
    Nlambda = max([int(f.split('params')[1].split('.pth')[0])
                   for f in os.listdir(result_dir) if 'params' in f]) + 1

    print(f"Loading regularization path of length {Nlambda}")

    params_dict = {i: torch.load(os.path.join(result_dir, f"params{i}.pth"),
                              map_location=torch.device('cpu')) for i in range(Nlambda)}

    regularization_strengths = [params_dict[i]['lam'].item() for i in range(Nlambda)]
    weights = [params_dict[i]['weight'] for i in range(Nlambda)]
    biases = [params_dict[i]['bias'] for i in range(Nlambda)]

    metrics = {'acc_tr': [], 'acc_val': [], 'acc_test': []}

    for k in metrics.keys():
        for i in range(Nlambda):
            metrics[k].append(params_dict[i]['metrics'][k])
        metrics[k] = 100 * np.stack(metrics[k])
    metrics = pd.DataFrame(metrics)
    metrics = metrics.rename(columns={'acc_tr': 'acc_train'})

    # weights_stacked = ch.stack(weights)
    # sparsity = ch.sum(weights_stacked != 0, dim=2).numpy()
    sparsity = np.array([torch.sum(w != 0, dim=1).numpy() for w in weights])

    return {'metrics': metrics,
            'regularization_strengths': regularization_strengths,
            'weights': weights,
            'biases': biases,
            'sparsity': sparsity,
            'weight_dense': weights[-1],
            'bias_dense': biases[-1]}
