"""Utilitários gerais: reprodutibilidade, dispositivo e ponte de parâmetros com o Flower."""

from __future__ import annotations

import os
import random
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm


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


def batchnorm_state_keys(model: nn.Module) -> set[str]:
    """Nomes (state_dict) dos parâmetros e buffers de todas as camadas BatchNorm.

    Usado pelo FedBN (Li et al. 2021): essas entradas ficam **locais** a cada
    cliente e não são sobrescritas pelo modelo global.
    """
    keys: set[str] = set()
    for mod_name, module in model.named_modules():
        if isinstance(module, _BatchNorm):
            prefix = f"{mod_name}." if mod_name else ""
            for pname, _ in module.named_parameters(recurse=False):
                keys.add(prefix + pname)
            for bname, _ in module.named_buffers(recurse=False):
                keys.add(prefix + bname)
    return keys


def set_parameters(
    model: nn.Module,
    parameters: list[np.ndarray],
    skip_keys: set[str] | None = None,
) -> None:
    """Carrega no modelo os pesos recebidos do servidor Flower.

    `skip_keys`: entradas do state_dict a **não** sobrescrever (mantém o valor
    local). Usado pelo FedBN para preservar as camadas BatchNorm de cada cliente.
    """
    skip = skip_keys or set()
    current = model.state_dict()
    state_dict = OrderedDict()
    for key, value in zip(current.keys(), parameters, strict=True):
        state_dict[key] = current[key] if key in skip else torch.as_tensor(value)
    model.load_state_dict(state_dict, strict=True)
