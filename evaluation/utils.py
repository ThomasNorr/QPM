import torch
from tqdm import tqdm



def get_metrics_for_model(train_loader, test_loader, model, metric_evaluator):
    (features_train, feature_maps_train, outputs_train, features_test, feature_maps_test,
     outputs_test, labels) = [], [], [], [], [], [], []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()
    model = model.to(device)
    training_transforms = train_loader.dataset.transform
    train_loader.dataset.transform = test_loader.dataset.transform # Use test transform for train
    train_loader = torch.utils.data.DataLoader(train_loader.dataset, batch_size=100, shuffle=False) # Turn off shuffling
    print("Going in get metrics")
    linear_matrix = model.linear.weight
    entries = torch.nonzero(linear_matrix)
    rel_features = torch.unique(entries[:, 1])
    with torch.no_grad():
        iterator = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in iterator:
            xs1 = data.to("cuda")
            output, feature_maps,  final_features = model(xs1,   with_feature_maps=True,                                                                                                              with_final_features=True,)
            outputs_train.append(output.to("cpu"))
            features_train.append(final_features.to("cpu"))
            labels.append(target.to("cpu"))
        total = 0
        correct = 0
        iterator = tqdm(enumerate(test_loader), total=len(test_loader))
        for batch_idx, (data, target) in iterator:
            xs1 = data.to("cuda")
            output, feature_maps, final_features = model(xs1, with_feature_maps=True,
                                                         with_final_features=True, )
            feature_maps_test.append(feature_maps[:, rel_features].to("cpu"))
            outputs_test.append(output.to("cpu"))
            total += target.size(0)
            _, predicted = output.max(1)
            correct += predicted.eq(target.to("cuda")).sum().item()
        print("test accuracy: ", correct / total)
        features_train = torch.cat(features_train)
        outputs_train = torch.cat(outputs_train)
        feature_maps_test = torch.cat(feature_maps_test)
        outputs_test = torch.cat(outputs_test)
        labels = torch.cat(labels)
        linear_matrix = linear_matrix[:, rel_features]
    print("Shape of linear matrix: ", linear_matrix.shape)
    all_metrics_dict = metric_evaluator(features_train,  outputs_train,
                                             feature_maps_test,
                                             outputs_test, linear_matrix,  labels)
    result_dict = {"Accuracy": correct / total,  "NFfeatures": linear_matrix.shape[1],
                   "PerClass": torch.nonzero(linear_matrix).shape[0] / linear_matrix.shape[0],
                   }
    result_dict.update(all_metrics_dict)
    print(result_dict)
    # Reset Train transforms
    train_loader.dataset.transform = training_transforms
    return result_dict
