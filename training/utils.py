
#from robustness.robustness.tools.helpers  https://github.com/MadryLab/robustness
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class VariableLossLogPrinter():
    def __init__(self):
        self.losses = {}

    def log_loss(self, key, val, n=1):
        if not key in self.losses:
            self.losses[key] = AverageMeter()
        self.losses[key].update(val, n)

    def get_loss_string(self):
        loss_string = " | ".join([f"{key}: {self.losses[key].avg:.4f}" for key in self.losses])

        return loss_string
