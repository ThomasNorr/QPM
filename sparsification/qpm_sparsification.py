import os
import sys

import numpy as np
import torch.utils.data

from sparsification.qpm.qpm_solving import solve_qp
from sparsification.qpm_constants.compute_A import compute_feat_class_corr_matrix
from sparsification.qpm_constants.compute_B import compute_locality_bias
from sparsification.qpm_constants.compute_R import compute_cos_sim_matrix
from sparsification.utils import get_feature_loaders

def compute_qpm_feature_selection_and_assignment(model, train_loader, test_loader, log_dir, n_classes, seed, n_features, per_class):
    feature_loaders, metadata, device,args =  get_feature_loaders(seed, log_dir,train_loader, test_loader, model, n_classes, )
    full_train_dataset = torch.utils.data.ConcatDataset([feature_loaders['train'].dataset, feature_loaders['val'].dataset])
    full_train_dataset_loader = torch.utils.data.DataLoader(full_train_dataset, batch_size=feature_loaders['train'].batch_size, shuffle=False, # Shuffling does not matter here
                                                            num_workers=feature_loaders['train'].num_workers)
    save_folder = log_dir / "qpm_constants_saved"
    save_folder.mkdir(parents=True, exist_ok=True)
    if  os.path.exists(save_folder / "A.pt") and os.path.exists(save_folder / "R.pt") and os.path.exists(save_folder / "B.pt"):
        a_matrix = torch.load(save_folder / "A.pt",map_location=torch.device('cpu') )
        r_matrix = torch.load(save_folder / "R.pt",map_location=torch.device('cpu') )
        b = torch.load(save_folder / "B.pt",map_location=torch.device('cpu') )
    else:
        a_matrix = compute_feat_class_corr_matrix(full_train_dataset_loader)
        a_matrix =  a_matrix / np.abs(a_matrix).max()
        r_matrix = compute_cos_sim_matrix(a_matrix)
        r_matrix = r_matrix / r_matrix.abs().max()
        b = compute_locality_bias(train_loader, model)

        torch.save(a_matrix, save_folder / "A.pt")
        torch.save(r_matrix, save_folder / "R.pt")
        torch.save(b, save_folder / "B.pt")
    r_matrix = torch.triu(torch.tensor(r_matrix))
    r_matrix[r_matrix < 0] = 0
    if  os.path.exists(save_folder / "sel.pt") and os.path.exists(save_folder / "weight.pt"):
        feature_sel = torch.load(save_folder / "sel.pt",map_location=torch.device('cpu') )
        weight = torch.load(save_folder / "weight.pt",map_location=torch.device('cpu') )
    else:
        feature_sel, weight = solve_qp(np.array(a_matrix),np.array(r_matrix), np.array(b), n_features, per_class, save_folder=save_folder)
        torch.save(feature_sel, save_folder / "sel.pt")
        torch.save(weight, save_folder / "weight.pt")
        if not torch.cuda.is_available():
            print("No GPU available, returning")
            sys.exit(0)
    mean, std = metadata["X"]['mean'], metadata["X"]['std']
    return feature_sel, weight.float(),  mean, std


