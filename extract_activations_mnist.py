import os
import random

import lucent.optvis.render as render
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
import numpy as np
from train_models import Net


def get_all_layers(model, layers, X):
    hooks = [render.ModuleHook(getattr(model, layer)) for layer in layers]
    model(X)
    for hook in hooks:
        hook.close()
    return [hook.features for hook in hooks]


def select_patch(image_activation, patch_size):
    x_cor, y_cor = random.randint(0, patch_size), random.randint(0, patch_size)
    return image_activation[:, :, x_cor, y_cor]


def get_data(number_of_images):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    data = datasets.MNIST("./data", train=True, transform=transform)
    return torch.cat(
        [data[i][0][None, :] for i in range(min(len(data), number_of_images))], 0
    )


def get_activations(
    model,
    layers,
    batch_size=150,
    number_of_images=100000,
):
    for layer in layers:
        if not os.path.exists(f"activations/{layer}"):
            os.makedirs(f"activations/{layer}")
    images = get_data(number_of_images)
    total_num_pictures = len(images)
    pbar = tqdm(total=total_num_pictures)
    while images.shape[0] > 0:
        batch, images = images[:batch_size], images[batch_size:]
        activations = get_all_layers(model, layers, batch)
        for layer, layer_activations in zip(layers, activations):
            patch_size = layer_activations.shape[3] - 1
            layer_activations = map(
                lambda img_activation: select_patch(img_activation, patch_size),
                layer_activations.split(1),
            )
            layer_activations = torch.cat(list(layer_activations))
            torch.save(
                layer_activations,
                f"activations/{layer}/activations_{total_num_pictures - len(images)}.pt",
            )
        pbar.update(batch_size)


def main():
    random.seed(17)
    layers = [
        "conv1",
        "conv2",
    ]

    extact_activations(layers)
    concat_batches(layers)


def extact_activations(layers):
    model = Net()
    model.load_state_dict(torch.load("mnist_cnn.pt"))
    get_activations(model, layers)


def concat_batches(layers, folder="activations", save_location="activations/MNIST"):
    if not os.path.exists(save_location):
        os.makedirs(save_location)
    for layer in layers:
        batch_paths = os.listdir(f"{folder}/{layer}")
        activations = torch.cat(
            [torch.load(f"{folder}/{layer}/{file}") for file in batch_paths]
        )
        # torch.save(activations, f"{save_location}/{layer}.pt")
        np.save(f"{save_location}/{layer}.npy", activations.detach().numpy())


if __name__ == "__main__":
    main()
