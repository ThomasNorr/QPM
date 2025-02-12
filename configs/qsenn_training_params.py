from configs.sldd_training_params import OptimizationScheduler


class  QSENNScheduler(OptimizationScheduler):
    def get_params(self):
        params = super().get_params()
        if self.n_calls >= 2:
            params[0] = params[0] * 0.9**(self.n_calls-2)
        if 2 <= self.n_calls <= 4:
            params[-2] = 10# Change num epochs to 10 for iterative finetuning
        return params
