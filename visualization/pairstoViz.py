import numpy as np
import torch


def get_similar_classes(model):
    weight_matrix = model.linear.weight
    pairwise_similarities = weight_matrix @ weight_matrix.T
    pairwise_similarities = pairwise_similarities.cpu().detach().numpy()
    pairwise_similarities[np.eye(pairwise_similarities.shape[0], dtype=bool)] = 0
    pairwise_similarities = np.triu(pairwise_similarities)
    average_per_class = weight_matrix.sum(dim=1).cpu().detach().numpy()
    target_sims = average_per_class - 1
    if (pairwise_similarities == target_sims).sum() > 0:
        candidates = np.argwhere(pairwise_similarities == target_sims)
    else:
        raise ValueError("No 4 in pairwise similarities")
    return  candidates


def find_easier_interpretable_pairs(model,  train_loader, min_sim):
    classes_indices = get_similar_classes(model)
    class_sim_gt = train_loader.dataset.get_class_sim()
    classes_indices = [[x, y] for x, y in classes_indices if class_sim_gt[x, y] > min_sim]
    return classes_indices


def select_clearly_activating_separable_samples(model, images, label):
    # Selects images that are classified correctly, have high diversity@5 and high feature activations
    with torch.no_grad():
        images = torch.stack(images).to("cuda")
        model = model.to("cuda")
        class_features = model.linear.weight[label].nonzero().flatten().tolist()
        output, feature_maps, final_features = model(images, with_feature_maps=True,
                                                     with_final_features=True, )

        trues = output.argmax(dim=1) == label
        print("Acc for class: ", label,trues.sum().item() / trues.shape[0])
        rel_maps = feature_maps[:, class_features]
        rel_features = final_features[:, class_features]
        softmaxed_maps = torch.nn.functional.softmax(rel_maps.flatten(2,3), dim=2)
        max_per_pos = softmaxed_maps.max(dim=2)[0].sum(dim=1)
        feature_sum = rel_features.sum(dim=1)
        score_per_sample =  max_per_pos * feature_sum
        score_per_sample -= score_per_sample.min() +1
        score_per_sample *= trues
        max_indices = score_per_sample.argmax(dim=0)
    return images[max_indices].to("cpu")

