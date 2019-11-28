import torch

from torch import nn


def set_padding_mode_for_torch_compatibility(model):
    """
    Pytorch 1.1 fails to load pre-1.1 models because it expects the newly added padding_mode attribute.
    This function fixes the problem by adding the missing attribute with the default value.
    """
    modules = model.modules()
    for layer in modules:
        if isinstance(layer, nn.Conv2d):
            setattr(layer, 'padding_mode', 'zeros')


def load_model(model_path, *args, **kwargs):
    model = torch.load(model_path, *args, **kwargs)

    if torch.__version__ == '1.1.0':
        print('Setting padding mode in Conv2d for torch 1.1 compatibility!')
        set_padding_mode_for_torch_compatibility(model)

    set_n_classes_for_pre_aug5_simple_conv_net(model)

    return model
