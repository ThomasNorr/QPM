# order: lr,weight_decay, step_lr, step_lr_gamma
import math


class EvaluatedDict:
    def __init__(self, d, func):
        self.dict = d
        self.func = func

    def __getitem__(self, key):
        return self.dict[self.func(key)]

dense_params = EvaluatedDict({False: [0.005, 0.0005, 30, 0.4, 150],True: [None,None,None,None,None],}, lambda x: x == "ImageNet")
def calculate_lr_from_args( epochs, step_lr, start_lr, step_lr_decay):
    # Gets the final learning rate after dense training with step_lr_schedule.
    n_steps = math.floor((epochs - step_lr) / step_lr)
    final_lr = start_lr * step_lr_decay ** n_steps
    return final_lr

ft_params =EvaluatedDict({False: [1e-4, 0.0005, 10, 0.4, 40],True:[[calculate_lr_from_args(150,30,0.005, 0.4), 0.0005, 10, 0.4, 40]]}, lambda x: x == "ImageNet")


