"""Utilitários gerais: reprodutibilidade, dispositivo e ponte de parâmetros com o Flower."""

from __future__ import annotations

import os
import random
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Fixa as sementes de random / numpy / torch para reprodutibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Extrai os pesos do modelo como lista de arrays NumPy (formato do Flower)."""
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Carrega no modelo os pesos recebidos do servidor Flower."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.as_tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
