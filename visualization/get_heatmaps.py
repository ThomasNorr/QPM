import cv2
import torch
import numpy as np
from visualization.colormaps import get_default_cmaps


def gamma_saturation(weights, gamma):
    initial_max = weights.max(axis=(1, 2), keepdims=True)
    weights = (weights ** gamma) / (weights.max() ** gamma)
    weights = weights * initial_max
    return weights

def get_visualizations(feature_indices, relevant_images,unnormalized_images, model,  gamma=1, norm_across_images=True,
                       scale=0.5):
    visualizations = []
    colormaps = get_default_cmaps()
    if norm_across_images:
        scale = 0.7
    for j, idx in enumerate(feature_indices):
        grayscale_cam = distribute_feature_maps(model, relevant_images, int(idx),
                                                             norm_across_images=norm_across_images)
        grayscale_cam = gamma_saturation(grayscale_cam, gamma)
        single_feature_line = overlay_images(unnormalized_images.cpu(), grayscale_cam, cmap=colormaps[j], scale=scale,
                                                 gray_scale_img=True)
        visualizations.append(torch.stack(single_feature_line))
    return visualizations

def overlay_images(relevant_images, grayscale_cam, cmap=cv2.COLORMAP_JET, scale=None, gray_scale_img=False):
    single_feature_line = []
    for i, rgb_img in enumerate(relevant_images):
        rgb_img = rgb_img.numpy().transpose(2, 1, 0)
        if gray_scale_img:
            rgb_img = rgb2gray(rgb_img)
        single_feature_line.append(torch.tensor(
            show_cam_on_image(rgb_img, grayscale_cam[i], use_rgb=True,
                              colormap=cmap, scale=scale).transpose(
                2,
                1,
                0)))
    return single_feature_line



def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

### From pytorch_grad_cam
def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET, scale=None) -> np.ndarray:
    """ This function overlays the cam mask on the image as an heatmap.
    By default the heatmap is in BGR format.

    :param img: The base image in RGB or BGR format.
    :param mask: The cam mask.
    :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
    :param colormap: The OpenCV colormap to be used.
    :returns: The default image with the cam overlay.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    if np.max(img) > 1:
        raise Exception(
            "The input image should np.float32 in the range [0, 1]")
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if scale is None:
        cam = heatmap + img
    else:
        cam = heatmap * scale + img * (1 - scale)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def distribute_feature_maps(model, images, index,  scale=True,
                            norm_across_images=False):
    """

    Args:
        model:
        images:
        index: Index of the output features

    Returns:

    """
    cuda = torch.cuda.is_available()
    if cuda:
        images = images.to("cuda")
        model = model.to("cuda")
    with torch.no_grad():
        features, featuremaps = model(images, with_feature_maps=True)
        featuremap = featuremaps[:, index]
    cam_map = featuremap
    cam_map = np.array(cam_map.cpu())
    if scale:
        scaled = scale_cam_image(cam_map, get_target_width_height(images))
    if norm_across_images and len(cam_map) > 1:
        mean_maps = featuremap.mean(axis=(1, 2)).cpu().numpy()
        scale_factor = torch.tensor(mean_maps[1] / mean_maps[0])
        print("Means when Scaling", mean_maps, "Scale ", scale_factor)
        maxs_init = np.sqrt(mean_maps)
        scaled *= maxs_init[:, None, None]
        scaled /= np.max(maxs_init)
    else:
        scaled = cam_map
        scaled = [cv2.resize(x, get_target_width_height(images)) for x in scaled]
    scaled = np.float32(scaled)
    scaled = np.transpose(scaled, (0, 2, 1))
    return scaled


def scale_cam_image(cam, target_size=None, scale_val=None):
    result = []
    for img in cam:
        img = img - np.min(img)
        max_val = np.max(img)
        img = img / (1e-7 + max_val)
        if scale_val is not None:
            img *= scale_val
        if target_size is not None:
            if len(img.shape) == 3:
                new_img = np.zeros((img.shape[0], target_size[1], target_size[0]))
                for i in range(img.shape[0]):
                    new_img[i] = cv2.resize(img[i], target_size)
                img = new_img
            else:
                img = cv2.resize(img, target_size)
        result.append(img)
    result = np.float32(result)

    return result

def get_target_width_height(input_tensor):
    width, height = input_tensor.size(-1), input_tensor.size(-2)
    return width, height
