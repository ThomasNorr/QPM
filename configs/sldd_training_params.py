from configs.optim_params import dense_params, ft_params


class OptimizationScheduler:
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_calls =  0


    def get_params(self):
        if self.n_calls == 0: # Return Deńse Params
            params =  dense_params[self.dataset]+ [False]
        else: # Return Finetuning Params
            params =  ft_params[self.dataset]+ [True]
        self.n_calls += 1
        return params

