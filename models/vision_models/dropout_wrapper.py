"""Dropout implementation as a wrapper class. The dropout layout is based on
https://github.com/google/uncertainty-baselines"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

import re
from dataclasses import dataclass
from typing import Callable
from torch import Tensor

from torch.nn import Module
from torch.nn.utils import parametrize

"""
Contains base wrapper classes
"""

import torch
from torch import nn


class ModelWrapper(nn.Module):
    """General model wrapper base class."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def set_grads(
        self,
        backbone_requires_grad: bool = True,
        classifier_requires_grad: bool = True,
        wrapper_requires_grad: bool = False,
    ):
        # Freeze / unfreeeze parts of the model:
        # backbone, classifier head and bias predictor head
        params_classifier = [p for p in self.model.get_classifier().parameters()]
        params_classifier_plus_backbone = [p for p in self.model.parameters()]
        params_classifier_plus_backbone_plus_wrapper = [p for p in self.parameters()]

        for param in params_classifier_plus_backbone_plus_wrapper:
            param.requires_grad = wrapper_requires_grad
        for param in params_classifier_plus_backbone:
            param.requires_grad = backbone_requires_grad
        for param in params_classifier:
            param.requires_grad = classifier_requires_grad

    def __getattr__(self, name: str):
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        return getattr(self.model, name)

    def forward_head(self, x, pre_logits: bool = False):
        # Always get pre_logits
        features = self.model.forward_head(x, pre_logits=True)

        if pre_logits:
            return features

        out = self.get_classifier()(features)

        if self.training:
            return out
        else:
            return {"logit": out, "feature": features}

    @staticmethod
    def convert_state_dict(state_dict):
        """
        Convert state_dict by removing 'model.' prefix from keys.
        """
        converted_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                converted_state_dict[k[6:]] = v  # Remove 'model.' prefix
            else:
                converted_state_dict[k] = v
        return converted_state_dict

    def load_model(self):
        """
        Load the model.
        """
        weight_path = self.weight_path
        checkpoint = torch.load(weight_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        state_dict = self.convert_state_dict(state_dict)

        self.model.load_state_dict(state_dict, strict=True)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)

        return x


class PosteriorWrapper(ModelWrapper):
    pass


class SpecialWrapper(ModelWrapper):
    pass


@dataclass
class ModuleData:
    variable_name: str
    module_name: str
    module: Module


def deep_setattr(obj, attr_path, value):
    parts = attr_path.split(".")

    for part in parts[:-1]:
        if part.isdigit():
            obj = obj[int(part)]
        else:
            obj = getattr(obj, part)

    if parts[-1].isdigit():
        obj[int(parts[-1])] = value
    else:
        setattr(obj, parts[-1], value)


def replace(model: Module, source_regex: str, target_module: Module):
    source_regex = re.compile(source_regex)

    module_datas = [
        ModuleData(
            variable_name=name,
            module_name=module.__class__.__name__,
            module=module,
        )
        for name, module in model.named_modules()
    ]
    matched_module_datas = [
        module_data
        for module_data in module_datas
        if source_regex.match(module_data.module_name)
    ]

    for matched_module_data in matched_module_datas:
        deep_setattr(
            model,
            matched_module_data.variable_name,
            target_module(matched_module_data.module),
        )


def replace_cond(
    model: Module,
    source_regex: str,
    cond: Callable[[Module], bool],
    target_module_true: Module,
    target_module_false: Module,
):
    source_regex = re.compile(source_regex)

    module_datas = [
        ModuleData(
            variable_name=name,
            module_name=module.__class__.__name__,
            module=module,
        )
        for name, module in model.named_modules()
    ]
    matched_module_datas = [
        module_data
        for module_data in module_datas
        if source_regex.match(module_data.module_name)
    ]

    for matched_module_data in matched_module_datas:
        target_module = (
            target_module_true
            if cond(matched_module_data.module)
            else target_module_false
        )
        deep_setattr(
            model,
            matched_module_data.variable_name,
            target_module(matched_module_data.module),
        )


def register(
    model: Module,
    source_regex: str,
    attribute_name: str,
    target_parametrization: Module,
):
    source_regex = re.compile(source_regex)

    module_datas = [
        ModuleData(
            variable_name=name,
            module_name=module.__class__.__name__,
            module=module,
        )
        for name, module in model.named_modules()
    ]
    matched_module_datas = [
        module_data
        for module_data in module_datas
        if source_regex.match(module_data.module_name)
    ]

    for matched_module_data in matched_module_datas:
        module = matched_module_data.module
        module_name = matched_module_data.module_name
        weight = getattr(module, attribute_name, None)
        if not isinstance(weight, Tensor):
            raise ValueError(
                f"Module '{module_name}' has no parameter or buffer with name "
                f"'{attribute_name}'"
            )

        parametrize.register_parametrization(
            module, attribute_name, target_parametrization(module=module), unsafe=True
        )


def register_cond(
    model: Module,
    source_regex: str,
    attribute_name: str,
    cond: Callable[[Module], bool],
    target_parametrization_true: Module,
    target_parametrization_false: Module,
):
    source_regex = re.compile(source_regex)

    module_datas = [
        ModuleData(
            variable_name=name,
            module_name=module.__class__.__name__,
            module=module,
        )
        for name, module in model.named_modules()
    ]
    matched_module_datas = [
        module_data
        for module_data in module_datas
        if source_regex.match(module_data.module_name)
    ]

    for matched_module_data in matched_module_datas:
        module = matched_module_data.module
        module_name = matched_module_data.module_name
        weight = getattr(module, attribute_name, None)
        if not isinstance(weight, Tensor):
            raise ValueError(
                f"Module '{module_name}' has no parameter or buffer with name "
                f"'{attribute_name}'"
            )

        target_parametrization = (
            target_parametrization_true
            if cond(matched_module_data.module)
            else target_parametrization_false
        )

        parametrize.register_parametrization(
            module, attribute_name, target_parametrization(module=module), unsafe=True
        )


class ActivationDropout(nn.Module):
    def __init__(self, dropout_probability, is_filterwise_dropout, activation):
        super().__init__()
        self.activation = activation
        dropout_function = F.dropout2d if is_filterwise_dropout else F.dropout
        self.dropout = partial(dropout_function, p=dropout_probability, training=True)

    def forward(self, inputs):
        x = self.activation(inputs)
        x = self.dropout(x)
        return x


class DropoutWrapper(PosteriorWrapper):
    """
    This module takes a model as input and creates a Dropout model from it.
    """

    def __init__(
        self,
        model: nn.Module,
        dropout_probability: float,
        is_filterwise_dropout: bool,
        num_mc_samples: int,
    ):
        super().__init__(model)

        self.num_mc_samples = num_mc_samples

        replace(
            model,
            "ReLU",
            partial(ActivationDropout, dropout_probability, is_filterwise_dropout),
        )
        replace(
            model,
            "GELU",
            partial(ActivationDropout, dropout_probability, is_filterwise_dropout),
        )

    def forward(self, inputs):
        if self.training:
            return self.model(inputs)  # [B, C]

        sampled_features = []
        sampled_logits = []
        for _ in range(self.num_mc_samples):
            features = self.model.forward_head(
                self.model.forward_features(inputs), pre_logits=True
            )
            logits = self.model(inputs)  # [B, C]

            sampled_features.append(features)
            sampled_logits.append(logits)

        sampled_features = torch.stack(sampled_features, dim=1)  # [B, S, D]
        mean_features = sampled_features.mean(dim=1)
        sampled_logits = torch.stack(sampled_logits, dim=1)  # [B, S, C]

        return {"logit": sampled_logits, "feature": mean_features}

    def forward_features(self, inputs):
        raise ValueError(f"forward_features cannot be called directly for {type(self)}")

    def forward_head(self, features):
        raise ValueError(f"forward_head cannot be called directly for {type(self)}")