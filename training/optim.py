from torch.optim import SGD, lr_scheduler

from configs.qsenn_training_params import QSENNScheduler
from configs.sldd_training_params import OptimizationScheduler
from training.img_net import get_default_img_schedule, get_default_img_optimizer


def get_optimizer(model,  schedulingClass):
    lr,weight_decay, step_lr, step_lr_gamma, n_epochs, finetune = schedulingClass.get_params()
    print("Optimizer LR set to ", lr)
    if lr is None: # Dense Training on ImageNet
        print("Learning rate is None, using Default Recipe for Resnet50")
        default_img_optimizer = get_default_img_optimizer(model)
        default_img_schedule = get_default_img_schedule(default_img_optimizer)
        return default_img_optimizer, default_img_schedule, 600
    if finetune:
        param_list = [x for x in model.parameters() if x.requires_grad]
    else:
        param_list = model.parameters()


    if finetune:
        optimizer = SGD(param_list,lr, momentum=0.95,
                        weight_decay=weight_decay)
    else:
        classifier_params_name = ["linear.bias","linear.weight"]
        classifier_params = [x[1] for x in
                             list(filter(lambda kv: kv[0] in classifier_params_name, model.named_parameters()))]
        base_params = [x[1] for x in list(
            filter(lambda kv: kv[0] not in classifier_params_name, model.named_parameters()))]

        optimizer = SGD([
            {'params': base_params},
            {"params": classifier_params, 'lr': 0.01}
        ], momentum=0.9, lr=lr, weight_decay=weight_decay)
    # Make schedule
    schedule = lr_scheduler.StepLR(optimizer, step_size=step_lr, gamma=step_lr_gamma)
    return optimizer, schedule, n_epochs


def get_scheduler_for_model(model, dataset):
    if model == "qsenn":
        return QSENNScheduler(dataset)
    else:
        return OptimizationScheduler(dataset)
