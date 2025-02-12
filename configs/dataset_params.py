import torch

from configs.optim_params import EvaluatedDict

dataset_constants = {"CUB2011":{"num_classes":200},
                      "TravelingBirds":{"num_classes":200},
                      "ImageNet":{"num_classes":1000},
                      "StanfordCars":{"num_classes":196},
                     "FGVCAircraft": {"num_classes":100}}

normalize_params = {"CUB2011":{"mean":  torch.tensor([0.4853, 0.4964, 0.4295]),"std":torch.tensor([0.2300, 0.2258, 0.2625])},
"TravelingBirds":{"mean":  torch.tensor([0.4584, 0.4369, 0.3957]),"std":torch.tensor([0.2610, 0.2569, 0.2722])},
                    "ImageNet":{'mean': torch.tensor([0.485, 0.456, 0.406]),'std': torch.tensor([0.229, 0.224, 0.225])} ,
"StanfordCars":{'mean': torch.tensor([0.4593, 0.4466, 0.4453]),'std': torch.tensor([0.2920, 0.2910, 0.2988])} ,
                  "FGVCAircraft":{'mean': torch.tensor([0.4827, 0.5130, 0.5352]),
                    'std': torch.tensor([0.2236, 0.2170, 0.2478]),}
                    }


dense_batch_size =  EvaluatedDict({False: 16,True: 1024,}, lambda x: x == "ImageNet")

ft_batch_size =  EvaluatedDict({False: 16,True: 1024,}, lambda x: x == "ImageNet")# Untested