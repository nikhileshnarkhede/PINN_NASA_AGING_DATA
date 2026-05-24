"""
models/backbone.py
==================
Generic neural network builder for IGBT RUL estimation.

The network is:  recurrent stack  ->  dense head  ->  single RUL output.

Every part of the architecture is declared as data in config.py -- the number
of recurrent layers, their hidden size and inter-layer dropout, and the dense
head's hidden widths, per-layer activations and per-layer dropout. To change
the architecture you edit config.py only; you never touch this file.

Self-describing checkpoints
---------------------------
The architecture is captured in an "arch spec" dict (current_arch_spec()).
The trainer saves this dict inside every checkpoint, and models are rebuilt
from the SAVED spec -- not from whatever config.py currently says. This means
a checkpoint always loads into the exact architecture it was trained with,
even after config.py has been edited. build_backbone_from_spec() is the
spec-driven builder; build_backbone() is a thin wrapper that uses the current
config's spec.

Paper-exact architecture (Lu, Guo, Liu & Shi 2023, Sci. Rep. 13:10167, Fig. 1)
is reproduced by:  RECURRENT_LAYERS = 1, DENSE_HIDDEN = [10],
                   DENSE_ACTIVATION = ["tanh"].

Two constraints, both fixed by what the model is -- not project choices:
  * The recurrent activation is set by the cell type (RNN = tanh, LSTM = its
    internal gates) and cannot be configured -- a PyTorch limitation.
  * The dense head always ends in a Linear(-> 1) output layer, appended
    automatically, because the model must emit exactly one RUL value.
"""
import logging

import torch
import torch.nn as nn

from config.config import (
    BACKBONE, N_FEATURES,
    RECURRENT_HIDDEN, RECURRENT_LAYERS, RECURRENT_DROPOUT,
    DENSE_HIDDEN, DENSE_ACTIVATION, DENSE_DROPOUT,
)

log = logging.getLogger(__name__)

# Supported dense-head activations. Add a new entry here to extend the set.
_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu":       nn.ReLU,
    "tanh":       nn.Tanh,
    "sigmoid":    nn.Sigmoid,
    "gelu":       nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "elu":        nn.ELU,
    "identity":   nn.Identity,
    "none":       nn.Identity,
}


# ----------------------------------------------------------------------------
# Architecture spec
# ----------------------------------------------------------------------------
def current_arch_spec() -> dict:
    """
    Snapshot the architecture currently described in config.py into a dict.

    The dict fully determines a model: it is saved into checkpoints by the
    trainer and replayed by build_backbone_from_spec() so a checkpoint always
    rebuilds into the architecture it was trained with.

    Returns:
        A dict with keys: backbone, n_features, recurrent_hidden,
        recurrent_layers, recurrent_dropout, dense_hidden, dense_activation,
        dense_dropout.
    """
    return {
        "backbone":          BACKBONE,
        "n_features":        N_FEATURES,
        "recurrent_hidden":  RECURRENT_HIDDEN,
        "recurrent_layers":  RECURRENT_LAYERS,
        "recurrent_dropout": RECURRENT_DROPOUT,
        "dense_hidden":      list(DENSE_HIDDEN),
        "dense_activation":  list(DENSE_ACTIVATION),
        "dense_dropout":     list(DENSE_DROPOUT),
    }


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _make_activation(name: str) -> nn.Module:
    """
    Build an activation module from its name.

    Args:
        name: Activation name (case-insensitive); see _ACTIVATIONS.

    Returns:
        The instantiated activation module.

    Raises:
        ValueError: If `name` is not a supported activation.
    """
    key = name.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"unknown activation '{name}'; choose from {list(_ACTIVATIONS)}"
        )
    return _ACTIVATIONS[key]()


def _validate_spec(spec: dict) -> None:
    """
    Fail fast on an inconsistent architecture spec.

    Args:
        spec: An architecture spec dict (see current_arch_spec).

    Raises:
        ValueError: If the dense-head lists have mismatched lengths, or
                    recurrent_layers < 1.
    """
    n = len(spec["dense_hidden"])
    if len(spec["dense_activation"]) != n:
        raise ValueError(
            f"dense_activation has {len(spec['dense_activation'])} entries "
            f"but dense_hidden has {n}; they must be the same length."
        )
    if len(spec["dense_dropout"]) != n:
        raise ValueError(
            f"dense_dropout has {len(spec['dense_dropout'])} entries "
            f"but dense_hidden has {n}; they must be the same length."
        )
    if spec["recurrent_layers"] < 1:
        raise ValueError(
            f"recurrent_layers must be >= 1, got {spec['recurrent_layers']}"
        )


def _build_dense_head(
    in_features: int,
    dense_hidden: list[int],
    dense_activation: list[str],
    dense_dropout: list[float],
) -> nn.Sequential:
    """
    Build the dense head from explicit spec lists.

    For each hidden layer i: Linear -> activation[i] -> (optional) Dropout[i].
    A final Linear(-> 1) output layer is always appended.

    Args:
        in_features:      Width of the recurrent stack's output.
        dense_hidden:     Hidden layer widths.
        dense_activation: Activation name per hidden layer.
        dense_dropout:    Dropout probability per hidden layer.

    Returns:
        An nn.Sequential mapping (batch, in_features) -> (batch, 1).
    """
    layers: list[nn.Module] = []
    prev = in_features

    for width, activation, dropout in zip(
            dense_hidden, dense_activation, dense_dropout):
        linear = nn.Linear(prev, width)
        nn.init.xavier_uniform_(linear.weight)
        nn.init.zeros_(linear.bias)
        layers.append(linear)
        layers.append(_make_activation(activation))
        if dropout and dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        prev = width

    output = nn.Linear(prev, 1)              # output layer -- width 1, automatic
    nn.init.xavier_uniform_(output.weight)
    nn.init.zeros_(output.bias)
    layers.append(output)

    return nn.Sequential(*layers)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
class RecurrentNet(nn.Module):
    """
    Many-to-one recurrent network: recurrent stack + dense head.

    Built entirely from an architecture spec dict, so the same class serves
    every configuration and every saved checkpoint. The spec is stored on the
    instance as `.spec`.

    Input  : (batch, SEQ_LEN, n_features)
    Output : (batch,)
    """

    def __init__(self, spec: dict) -> None:
        """
        Args:
            spec: An architecture spec dict (see current_arch_spec).
        """
        super().__init__()
        _validate_spec(spec)
        self.spec = spec

        kind = spec["backbone"].lower()
        recurrent_cls = {"rnn": nn.RNN, "lstm": nn.LSTM}[kind]
        kwargs = dict(
            input_size=spec["n_features"],
            hidden_size=spec["recurrent_hidden"],
            num_layers=spec["recurrent_layers"],
            batch_first=True,
            # PyTorch applies inter-layer dropout only when num_layers > 1.
            dropout=(spec["recurrent_dropout"]
                     if spec["recurrent_layers"] > 1 else 0.0),
        )
        if kind == "rnn":
            kwargs["nonlinearity"] = "tanh"

        self.recurrent = recurrent_cls(**kwargs)
        self.head = _build_dense_head(
            spec["recurrent_hidden"], spec["dense_hidden"],
            spec["dense_activation"], spec["dense_dropout"])
        self._init_recurrent()

    def _init_recurrent(self) -> None:
        """Orthogonal init for recurrent weights, zeros for biases."""
        for name, param in self.recurrent.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input windows, shape (batch, SEQ_LEN, n_features).

        Returns:
            RUL predictions, shape (batch,).
        """
        out, _ = self.recurrent(x)
        return self.head(out[:, -1, :]).squeeze(-1)   # many-to-one: last step


# ----------------------------------------------------------------------------
# Factories
# ----------------------------------------------------------------------------
def build_backbone_from_spec(spec: dict) -> nn.Module:
    """
    Build a model from an explicit architecture spec.

    This is what checkpoint-loading code uses: pass the spec stored in the
    checkpoint and the model rebuilds exactly, independent of config.py.

    Args:
        spec: An architecture spec dict (see current_arch_spec).

    Returns:
        An nn.Module mapping (batch, SEQ_LEN, n_features) -> (batch,).

    Raises:
        ValueError: If the backbone is unknown or the spec is inconsistent.
    """
    kind = spec["backbone"].lower()
    if kind not in ("rnn", "lstm"):
        raise ValueError(f"unknown backbone '{kind}'; choose 'rnn' or 'lstm'")

    model = RecurrentNet(spec)
    log.info("built '%s' | %d recurrent layer(s) x %d hidden | dense head %s",
             kind, spec["recurrent_layers"], spec["recurrent_hidden"],
             list(spec["dense_hidden"]) + [1])
    log.info("  trainable parameters: %d", count_parameters(model))
    return model


def build_backbone(name: str | None = None) -> nn.Module:
    """
    Build the model described by the current config.py.

    Thin wrapper over build_backbone_from_spec using current_arch_spec(). This
    is the entry point for training and visualisation -- fresh models always
    follow config.py.

    Args:
        name: Backbone key ("rnn" | "lstm"); defaults to config.BACKBONE.

    Returns:
        An nn.Module mapping (batch, SEQ_LEN, n_features) -> (batch,).
    """
    spec = current_arch_spec()
    if name is not None:
        spec = dict(spec, backbone=name)
    return build_backbone_from_spec(spec)


def count_parameters(model: nn.Module) -> int:
    """
    Return the number of trainable parameters.

    Args:
        model: Any nn.Module.

    Returns:
        Count of parameters with requires_grad=True.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test: build each backbone from the current config.
    #   python -m models.backbone
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)-8s | %(message)s")

    dummy = torch.randn(8, 10, N_FEATURES)
    for arch in ("rnn", "lstm"):
        net = build_backbone(arch)
        net.eval()
        with torch.no_grad():
            y = net(dummy)
        log.info("%-5s | input %s -> output %s",
                 arch, tuple(dummy.shape), tuple(y.shape))
