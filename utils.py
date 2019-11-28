import os
import json
import logging
import shutil

import torch
import numpy as np

from util.model_loading_and_compatibility_util import load_model

def tensor2array(tensor):
    """
    Convert a torch tensor to numpy ndarray.

    Parameters
    ----------
    var : torch.Tensor
        Input, to be converted to numpy.

    Returns
    -------
    array : np.ndarray
        Numpy array.

    """
    if isinstance(tensor, np.ndarray):
        return tensor
    if tensor.is_cuda:
        tensor = tensor.to('cpu')
    if tensor.requires_grad:
        tensor = tensor.detach()
    tensor = tensor.numpy()
    if tensor.dtype == np.uint8:
        # convert byte tensor to actual boolean tensor.
        tensor = tensor.astype(bool)
    return tensor


def array2tensor(array, device=None, cuda=False):
    """
    Convert a numpy ndarray or torch tensor to torch variable.

    Parameters
    ----------
    array : np.ndarray or torch.Tensor
        Numpy array.

    Returns
    -------
    tensor : torch.Tensor
        Tensor.

    """
    if isinstance(array, torch.Tensor):  # note I checked this also works with CUDA tensors
        tensor = array
    else:
        tensor = torch.from_numpy(array)
    if device is not None:
        tensor = tensor.to(device)
    if cuda:
        tensor = tensor.cuda()
    return tensor
class Params:
    """Class that loads hyperparameters from a json file.
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage:
    """A simple class that maintains the running average of a quantity
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file
    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'
    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.
    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = load_model(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def load_checkpoint(path, use_cuda=False):
    if use_cuda:
        return load_model(path)
    else:
        return load_model(path, map_location=lambda storage, loc: storage)


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device("cpu")

    return device


def to_device(tensor_or_list, device):
    if isinstance(tensor_or_list, (list, tuple)):
        tensor_or_list = [tensor.to(device) for tensor in tensor_or_list]
    else:
        tensor_or_list = tensor_or_list.to(device)

    return tensor_or_list



def list_public_members(obj):
    """Return list of all the specified object's members supposedly for public access."""
    return [member for member in dir(obj) if member[0] != '_']


class NotListingNoneAttrs:
    """Class of objects to list only attributes that don't have value None"""

    # For those interested:
    # The use of this class follows from classes like DatasetArgs replacing a dictionary in previous implementation;
    # the dictionary keys present, depended on the run configuration; then the dictionary's items were
    # passed (through **dict) to the __init__ of a class, which accepted only exactly those keys. While this can
    # work as expected, it's an implicit and quite fragile construction.
    # This class is now used to imitate the **dict call. Better would be to make
    # explicit which arguments the receiving __init__ accepts, and pass in only those values.

    def vars(self):
        return {attr: val for attr, val in vars(self).items() if val is not None}
