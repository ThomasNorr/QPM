import os
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from FeatureDiversityLoss import FeatureDiversityLoss
from architectures.model_mapping import get_model
from configs.architecture_params import architecture_params
from configs.dataset_params import dataset_constants
from evaluation.qpm_metrics import eval_model_on_all_qpm_metrics
from evaluation.qsenn_metrics import eval_model_on_all_qsenn_metrics
from finetuning.map_function import finetune
from get_data import get_data
from saving.logging import Tee
from saving.utils import json_save
from train import train, test
from training.optim import get_optimizer, get_scheduler_for_model


def main(dataset, arch,seed=None, model_type="qsenn", do_dense=True,crop = True, n_features = 50, n_per_class=5, img_size=448, reduced_strides=False):
    # create random seed, if seed is None
    if seed is None:
        seed = np.random.randint(0, 1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    dataset_key = dataset
    if crop:
        assert dataset in ["CUB2011","TravelingBirds"]
        dataset_key += "_crop"
    log_dir = Path.home()/f"tmp/{arch}/{dataset_key}/{seed}/"
    log_dir.mkdir(parents=True, exist_ok=True)
    tee = Tee(log_dir / "log.txt") # save log to file
    n_classes = dataset_constants[dataset]["num_classes"]
    train_loader, test_loader = get_data(dataset, crop=crop, img_size=img_size)
    model = get_model(arch, n_classes, reduced_strides)
    fdl = FeatureDiversityLoss(architecture_params[arch]["beta"]  , model.linear)
    OptimizationSchedule = get_scheduler_for_model(model_type, dataset)
    optimizer, schedule, dense_epochs =get_optimizer(model,   OptimizationSchedule)
    if not os.path.exists(log_dir / "Trained_DenseModel.pth"):
        if do_dense:
            for epoch in trange(dense_epochs):
                model = train(model, train_loader, optimizer, fdl, epoch)
                schedule.step()
                if epoch % 5 == 0:
                    test(model, test_loader,epoch)
        else:
            print("Using pretrained model, only makes sense for ImageNet")
        torch.save(model.state_dict(), os.path.join(log_dir, f"Trained_DenseModel.pth"))
    else:
        model.load_state_dict(torch.load(log_dir / "Trained_DenseModel.pth", map_location="cpu" if not torch.cuda.is_available() else "cuda"))
    if not  os.path.exists( log_dir/f"Results_DenseModel.json"):
        metrics_dense = eval_model_on_all_qpm_metrics(model, test_loader, train_loader)
        json_save(os.path.join(log_dir, f"Results_DenseModel.json"), metrics_dense)
    final_model = finetune(model_type, model, train_loader, test_loader, log_dir, n_classes, seed, architecture_params[arch]["beta"], OptimizationSchedule, n_per_class, n_features)
    torch.save(final_model.state_dict(), os.path.join(log_dir,f"{model_type}_{n_features}_{n_per_class}_FinetunedModel.pth"))
    metrics_finetuned = eval_model_on_all_qpm_metrics(model, test_loader, train_loader)
    json_save(os.path.join(log_dir, f"Results_{model_type}_{n_features}_{n_per_class}_FinetunedModel.json"), metrics_finetuned)
    print("Done")
    pass



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', default="CUB2011", type=str, help='dataset name', choices=["CUB2011", "ImageNet", "TravelingBirds", "StanfordCars"])
    parser.add_argument('--arch', default="resnet50", type=str, help='Backbone Feature Extractor', choices=["resnet50","resnet34", "resnet18"])
    parser.add_argument('--model_type', default="qpm", type=str, help='Type of Model', choices=["qsenn", "sldd", "qpm"])
    parser.add_argument('--seed', default=None, type=int, help='seed, used for naming the folder and random processes. Could be useful to set to have multiple finetune runs (e.g. Q-SENN and SLDD) on the same dense model') # 769567, 552629
    parser.add_argument('--do_dense', default=True, type=bool, help='whether to train dense model. Should be true for all datasets except (maybe) ImageNet')
    parser.add_argument('--cropGT', default=True, type=bool,
                        help='Whether to crop CUB/TravelingBirds based on GT Boundaries')
    parser.add_argument('--n_features', default=50, type=int, help='How many features to select') #769567
    parser.add_argument('--n_per_class', default=5, type=int, help='How many features to assign to each class')
    parser.add_argument('--img_size', default=224, type=int, help='Image size')
    parser.add_argument('--reduced_strides', default=True, type=bool, help='Whether to use reduced strides for resnets')
    args = parser.parse_args()
    main(args.dataset, args.arch, args.seed, args.model_type, args.do_dense,args.cropGT,  args.n_features, args.n_per_class, args.img_size, args.reduced_strides)
