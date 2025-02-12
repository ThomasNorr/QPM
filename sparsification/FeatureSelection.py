from argparse import ArgumentParser
import logging
import math
import os.path
import sys
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from glm_saga.elasticnet import maximum_reg_loader, get_device, elastic_loss_and_acc_loader
from torch import nn

import torch as ch

from sparsification.utils import safe_zip

# TODO checkout this change: Marks changes to the group version of glmsaga

"""
This would need glm_saga to run
usage to select 50 features with parameters as in paper:
metadata contains information about the precomputed train features in feature_loaders
args contains the default arguments for glm-saga, as described at the bottom
def get_glm_to_zero(feature_loaders, metadata,  args, num_classes, device, train_ds, Ntotal):
    num_features = metadata["X"]["num_features"][0]
    fittingClass = FeatureSelectionFitting(num_features, num_lasses, args, 0.8,
                                           50,
                                           True,0.1,
                                           lookback=3, tol=1e-4,
                                           epsilon=1,)
    to_drop, test_acc = fittingClass.fit(feature_loaders, metadata, device)
    return to_drop

to_drop is then used to remove the features from the downstream fitting and finetuning.
"""


class FeatureSelectionFitting:
    def __init__(self, n_features, n_classes, args, selalpha, nKeep, lam_fac,out_dir, lookback=None, tol=None,
                 epsilon=None):
        """
        This is an adaption of the group version of glm-saga (https://github.com/MadryLab/DebuggableDeepNetworks)
        The function extended_mask_max covers the changed operator,
        Args:
            n_features:
            n_classes:
            args: default args for glmsaga
            selalpha: alpha for elastic net
            nKeep: target number features
            lam_fac: discount factor for lambda
            parameters of glmsaga
            lookback:
            tol:
            epsilon:
        """
        self.selected_features = torch.zeros(n_features, dtype=torch.bool)
        self.num_features = n_features
        self.selalpha = selalpha
        self.lam_Fac = lam_fac
        self.out_dir = out_dir
        self.n_classes = n_classes
        self.nKeep = nKeep
        self.args = self.extend_args(args, lookback, tol, epsilon)

    # Extended Proximal Operator for Feature Selection
    def extended_mask_max(self, greater_to_keep, thresh):
        prev = greater_to_keep[self.selected_features]
        greater_to_keep[self.selected_features] = torch.min(greater_to_keep)
        max_entry = torch.argmax(greater_to_keep)
        greater_to_keep[self.selected_features] = prev
        mask = torch.zeros_like(greater_to_keep)
        mask[max_entry] = 1
        final_mask = (greater_to_keep > thresh)
        final_mask = final_mask * mask
        allowed_to_keep = torch.logical_or(self.selected_features, final_mask)
        return allowed_to_keep

    def extend_args(self, args, lookback, tol, epsilon):
        for key, entry in safe_zip(["lookbehind", "tol",
                                    "lr_decay_factor", ], [lookback, tol, epsilon]):
            if entry is not None:
                setattr(args, key, entry)
        return args

    # Grouped L1 regularization
    # proximal operator for f(weight) = lam * \|weight\|_2
    # where the 2-norm is taken columnwise
    def group_threshold(self, weight, lam):
        norm = weight.norm(p=2, dim=0) + 1e-6
        #  print(ch.sum((norm > lam)))
        return (weight - lam * weight / norm) * self.extended_mask_max(norm, lam)

    # Elastic net regularization with group sparsity
    # proximal operator for f(x) = alpha * \|x\|_1 + beta * \|x\|_2^2
    # where the 2-norm is taken columnwise
    def group_threshold_with_shrinkage(self, x, alpha, beta):
        y = self.group_threshold(x, alpha)
        return y / (1 + beta)

    def threshold(self, weight_new, lr, lam):
        alpha = self.selalpha
        if alpha == 1:
            # Pure L1 regularization
            weight_new = self.group_threshold(weight_new, lr * lam * alpha)
        else:
            # Elastic net regularization
            weight_new = self.group_threshold_with_shrinkage(weight_new, lr * lam * alpha,
                                                             lr * lam * (1 - alpha))
        return weight_new

    # Train an elastic GLM with proximal SAGA
    # Since SAGA stores a scalar for each example-class pair, either pass
    # the number of examples and number of classes or calculate it with an
    # initial pass over the loaders
    def train_saga(self, linear, loader, lr, nepochs, lam, alpha, group=True, verbose=None,
                   state=None, table_device=None, n_ex=None, n_classes=None, tol=1e-4,
                   preprocess=None, lookbehind=None, family='multinomial', logger=None):
        if logger is None:
            logger = print
        with ch.no_grad():
            weight, bias = list(linear.parameters())
            if table_device is None:
                table_device = weight.device

            # get total number of examples and initialize scalars
            # for computing the gradients
            if n_ex is None:
                n_ex = sum(tensors[0].size(0) for tensors in loader)
            if n_classes is None:
                if family == 'multinomial':
                    n_classes = max(tensors[1].max().item() for tensors in loader) + 1
                elif family == 'gaussian':
                    for batch in loader:
                        y = batch[1]
                        break
                    n_classes = y.size(1)

            # Storage for scalar gradients and averages
            if state is None:
                a_table = ch.zeros(n_ex, n_classes).to(table_device)
                w_grad_avg = ch.zeros_like(weight).to(weight.device)
                b_grad_avg = ch.zeros_like(bias).to(weight.device)
            else:
                a_table = state["a_table"].to(table_device)
                w_grad_avg = state["w_grad_avg"].to(weight.device)
                b_grad_avg = state["b_grad_avg"].to(weight.device)

            obj_history = []
            obj_best = None
            nni = 0
            for t in range(nepochs):
                total_loss = 0
                for n_batch, batch in enumerate(loader):
                    if len(batch) == 3:
                        X, y, idx = batch
                        w = None
                    elif len(batch) == 4:
                        X, y, w, idx = batch
                    else:
                        raise ValueError(
                            f"Loader must return (data, target, index) or (data, target, index, weight) but instead got a tuple of length {len(batch)}")

                    if preprocess is not None:
                        device = get_device(preprocess)
                        with ch.no_grad():
                            X = preprocess(X.to(device))
                    X = X.to(weight.device)
                    out = linear(X)

                    # split gradient on only the cross entropy term
                    # for efficient storage of gradient information
                    if family == 'multinomial':
                        if w is None:
                            loss = F.cross_entropy(out, y.to(weight.device), reduction='mean')
                        else:
                            loss = F.cross_entropy(out, y.to(weight.device), reduction='none')
                            loss = (loss * w).mean()
                        I = ch.eye(linear.weight.size(0))
                        target = I[y].to(weight.device)  # change to OHE

                        # Calculate new scalar gradient
                        logits = F.softmax(linear(X))
                    elif family == 'gaussian':
                        if w is None:
                            loss = 0.5 * F.mse_loss(out, y.to(weight.device), reduction='mean')
                        else:
                            loss = 0.5 * F.mse_loss(out, y.to(weight.device), reduction='none')
                            loss = (loss * (w.unsqueeze(1))).mean()
                        target = y

                        # Calculate new scalar gradient
                        logits = linear(X)
                    else:
                        raise ValueError(f"Unknown family: {family}")
                    total_loss += loss.item() * X.size(0)

                    # BS x NUM_CLASSES
                    a = logits - target
                    if w is not None:
                        a = a * w.unsqueeze(1)
                    a_prev = a_table[idx].to(weight.device)

                    # weight parameter
                    w_grad = (a.unsqueeze(2) * X.unsqueeze(1)).mean(0)
                    w_grad_prev = (a_prev.unsqueeze(2) * X.unsqueeze(1)).mean(0)
                    w_saga = w_grad - w_grad_prev + w_grad_avg
                    weight_new = weight - lr * w_saga
                    weight_new = self.threshold(weight_new, lr, lam)
                    # bias parameter
                    b_grad = a.mean(0)
                    b_grad_prev = a_prev.mean(0)
                    b_saga = b_grad - b_grad_prev + b_grad_avg
                    bias_new = bias - lr * b_saga

                    # update table and averages
                    a_table[idx] = a.to(table_device)
                    w_grad_avg.add_((w_grad - w_grad_prev) * X.size(0) / n_ex)
                    b_grad_avg.add_((b_grad - b_grad_prev) * X.size(0) / n_ex)

                    if lookbehind is None:
                        dw = (weight_new - weight).norm(p=2)
                        db = (bias_new - bias).norm(p=2)
                        criteria = ch.sqrt(dw ** 2 + db ** 2)

                        if criteria.item() <= tol:
                            return {
                                "a_table": a_table.cpu(),
                                "w_grad_avg": w_grad_avg.cpu(),
                                "b_grad_avg": b_grad_avg.cpu()
                            }

                    weight.data = weight_new
                    bias.data = bias_new

                saga_obj = total_loss / n_ex + lam * alpha * weight.norm(p=1) + 0.5 * lam * (1 - alpha) * (
                        weight ** 2).sum()

                # save amount of improvement
                obj_history.append(saga_obj.item())
                if obj_best is None or saga_obj.item() + tol < obj_best:
                    obj_best = saga_obj.item()
                    nni = 0
                else:
                    nni += 1

                # Stop if no progress for lookbehind iterationsd:])
                criteria = lookbehind is not None and (nni >= lookbehind)

                nnz = (weight.abs() > 1e-5).sum().item()
                total = weight.numel()
                if verbose and (t % verbose) == 0:
                    if lookbehind is None:
                        logger(
                            f"obj {saga_obj.item()} weight nnz {nnz}/{total} ({nnz / total:.4f}) criteria {criteria:.4f} {dw} {db}")
                    else:
                        logger(
                            f"obj {saga_obj.item()} weight nnz {nnz}/{total} ({nnz / total:.4f}) obj_best {obj_best}")

                if lookbehind is not None and criteria:
                    logger(
                        f"obj {saga_obj.item()} weight nnz {nnz}/{total} ({nnz / total:.4f}) obj_best {obj_best} [early stop at {t}]")
                    return {
                        "a_table": a_table.cpu(),
                        "w_grad_avg": w_grad_avg.cpu(),
                        "b_grad_avg": b_grad_avg.cpu()
                    }

            logger(f"did not converge at {nepochs} iterations (criteria {criteria})")
            return {
                "a_table": a_table.cpu(),
                "w_grad_avg": w_grad_avg.cpu(),
                "b_grad_avg": b_grad_avg.cpu()
            }

    def glm_saga(self, linear, loader, max_lr, nepochs, alpha, dropout, tries,
                 table_device=None, preprocess=None, group=False,
                 verbose=None, state=None, n_ex=None, n_classes=None,
                 tol=1e-4, epsilon=0.001, k=100, checkpoint=None,
                 do_zero=True, lr_decay_factor=1, metadata=None,
                 val_loader=None, test_loader=None, lookbehind=None,
                 family='multinomial', encoder=None, tot_tries=1):
        if encoder is not None:
            warnings.warn("encoder argument is deprecated; please use preprocess instead", DeprecationWarning)
            preprocess = encoder
        device = get_device(linear)
        checkpoint = self.out_dir
        if preprocess is not None and (device != get_device(preprocess)):
            raise ValueError(
                f"Linear and preprocess must be on same device (got {get_device(linear)} and {get_device(preprocess)})")

        if metadata is not None:
            if n_ex is None:
                n_ex = metadata['X']['num_examples']
            if n_classes is None:
                n_classes = metadata['y']['num_classes']
        lam_fac = (1 + (tries - 1) / tot_tries)
        print("Using lam_fac ", lam_fac)
        max_lam = maximum_reg_loader(loader, group=group, preprocess=preprocess, metadata=metadata,
                                     family=family) / max(
            0.001, alpha) * lam_fac
        group_lam = maximum_reg_loader(loader, group=True, preprocess=preprocess, metadata=metadata,
                                       family=family) / max(
            0.001, alpha) * lam_fac
        min_lam = epsilon * max_lam
        group_min_lam = epsilon * group_lam
        # logspace is base 10 but log is base e so use log10
        lams = ch.logspace(math.log10(max_lam), math.log10(min_lam), k)
        lrs = ch.logspace(math.log10(max_lr), math.log10(max_lr / lr_decay_factor), k)
        found = False
        if do_zero:
            lams = ch.cat([lams, lams.new_zeros(1)])
            lrs = ch.cat([lrs, lrs.new_ones(1) * lrs[-1]])

        path = []
        best_val_loss = float('inf')

        if checkpoint is not None:
            os.makedirs(checkpoint, exist_ok=True)

            file_handler = logging.FileHandler(filename=os.path.join(checkpoint, 'output.log'))
            stdout_handler = logging.StreamHandler(sys.stdout)
            handlers = [file_handler, stdout_handler]

            logging.basicConfig(
                level=logging.DEBUG,
                format='[%(asctime)s] %(levelname)s - %(message)s',
                handlers=handlers
            )
            logger = logging.getLogger('glm_saga').info
        else:
            logger = print
        while self.selected_features.sum() < self.nKeep:  # TODO checkout this change, one iteration per feature
            n_feature_to_keep = self.selected_features.sum()
            for i, (lam, lr) in enumerate(zip(lams, lrs)):
                lam = lam * self.lam_Fac
                start_time = time.time()
                self.selected_features = self.selected_features.to(device)
                state = self.train_saga(linear, loader, lr, nepochs, lam, alpha,
                                        table_device=table_device, preprocess=preprocess, group=group, verbose=verbose,
                                        state=state, n_ex=n_ex, n_classes=n_classes, tol=tol, lookbehind=lookbehind,
                                        family=family, logger=logger)

                with ch.no_grad():
                    loss, acc = elastic_loss_and_acc_loader(linear, loader, lam, alpha, preprocess=preprocess,
                                                            family=family)
                    loss, acc = loss.item(), acc.item()

                    loss_val, acc_val = -1, -1
                    if val_loader:
                        loss_val, acc_val = elastic_loss_and_acc_loader(linear, val_loader, lam, alpha,
                                                                        preprocess=preprocess,
                                                                        family=family)
                        loss_val, acc_val = loss_val.item(), acc_val.item()

                    loss_test, acc_test = -1, -1
                    if test_loader:
                        loss_test, acc_test = elastic_loss_and_acc_loader(linear, test_loader, lam, alpha,
                                                                          preprocess=preprocess, family=family)
                        loss_test, acc_test = loss_test.item(), acc_test.item()

                    params = {
                        "lam": lam,
                        "lr": lr,
                        "alpha": alpha,
                        "time": time.time() - start_time,
                        "loss": loss,
                        "metrics": {
                            "loss_tr": loss,
                            "acc_tr": acc,
                            "loss_val": loss_val,
                            "acc_val": acc_val,
                            "loss_test": loss_test,
                            "acc_test": acc_test,
                        },
                        "weight": linear.weight.detach().cpu().clone(),
                        "bias": linear.bias.detach().cpu().clone()

                    }
                    path.append(params)
                    if loss_val is not None and loss_val < best_val_loss:
                        best_val_loss = loss_val
                        best_params = params
                        found = True
                    nnz = (linear.weight.abs() > 1e-5).sum().item()
                    total = linear.weight.numel()
                    if family == 'multinomial':
                        logger(
                            f"{n_feature_to_keep} Feature ({i}) lambda {lam:.4f}, loss {loss:.4f}, acc {acc:.4f} [val acc {acc_val:.4f}] [test acc {acc_test:.4f}], sparsity {nnz / total} [{nnz}/{total}], time {time.time() - start_time}, lr {lr:.4f}")
                    elif family == 'gaussian':
                        logger(
                            f"({i}) lambda {lam:.4f}, loss {loss:.4f} [val loss {loss_val:.4f}] [test loss {loss_test:.4f}], sparsity {nnz / total} [{nnz}/{total}], time {time.time() - start_time}, lr {lr:.4f}")

                if self.check_new_feature(linear.weight):  # TODO checkout this change, canceling if new feature is used
                    if checkpoint is not None:
                        ch.save(params, os.path.join(checkpoint, f"params{n_feature_to_keep}.pth"))
                    break
        if found:
            return {
                'path': path,
                'best': best_params,
                'state': state
            }
        else:
            return False

    def check_new_feature(self, weight):
        # TODO checkout this change, checking if new feature is used
        copied_weight = torch.tensor(weight.cpu())
        used_features = torch.unique(
            torch.nonzero(copied_weight)[:, 1])
        if len(used_features) > 0:
            new_set = set(used_features.tolist())
            old_set = set(torch.nonzero(self.selected_features)[:, 0].tolist())
            diff = new_set - old_set
            if len(diff) > 0:
                self.selected_features[used_features] = True
                return True
        return False

    def fit(self, feature_loaders, metadata, device):
        # TODO checkout this change, glm saga code slightly adapted to return to_drop
        print("Initializing linear model...")
        linear = nn.Linear(self.num_features, self.n_classes).to(device)
        for p in [linear.weight, linear.bias]:
            p.data.zero_()

        print("Preparing normalization preprocess and indexed dataloader")
        preprocess = NormalizedRepresentation(feature_loaders['train'],
                                              metadata=metadata,
                                              device=linear.weight.device)

        print("Calculating the regularization path")
        mpl_logger = logging.getLogger("matplotlib")
        mpl_logger.setLevel(logging.WARNING)
        selected_features = self.glm_saga(linear,
                                          feature_loaders['train'],
                                          self.args.lr,
                                          self.args.max_epochs,
                                          self.selalpha, 0, 1,
                                          val_loader=feature_loaders['val'],
                                          test_loader=feature_loaders['test'],
                                          n_classes=self.n_classes,
                                          verbose=self.args.verbose,
                                          tol=self.args.tol,
                                          lookbehind=self.args.lookbehind,
                                          lr_decay_factor=self.args.lr_decay_factor,
                                          group=True,
                                          epsilon=self.args.lam_factor,
                                          metadata=metadata,
                                          preprocess=preprocess, tot_tries=1)
        to_drop = np.where(self.selected_features.cpu().numpy() == 0)[0]
        test_acc = selected_features["path"][-1]["metrics"]["acc_test"]
        torch.set_grad_enabled(True)
        return to_drop, test_acc


class NormalizedRepresentation(ch.nn.Module):
    def __init__(self, loader, metadata, device='cuda', tol=1e-5):
        super(NormalizedRepresentation, self).__init__()

        assert metadata is not None
        self.device = device
        self.mu = metadata['X']['mean']
        self.sigma = ch.clamp(metadata['X']['std'], tol)

    def forward(self, X):
        return (X - self.mu.to(self.device)) / self.sigma.to(self.device)




