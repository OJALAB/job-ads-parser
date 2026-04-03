from __future__ import annotations

import re
import warnings
from dataclasses import dataclass


CUDA_DEVICE_PATTERN = re.compile(r"^cuda(?::(?P<index>\d+))?$")


@dataclass(frozen=True, slots=True)
class ResolvedDevice:
    requested: str
    resolved: str
    use_cpu: bool
    hf_device: int
    cuda_index: int | None


def _load_torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def resolve_device_spec(device: str) -> ResolvedDevice:
    requested = (device or "auto").strip().lower()
    if not requested:
        requested = "auto"

    torch = _load_torch()
    cuda_match = CUDA_DEVICE_PATTERN.fullmatch(requested)

    if requested == "cpu":
        return ResolvedDevice(
            requested=requested,
            resolved="cpu",
            use_cpu=True,
            hf_device=-1,
            cuda_index=None,
        )

    if requested == "auto":
        cuda_available = bool(torch is not None and torch.cuda.is_available())
        if cuda_available:
            return ResolvedDevice(
                requested=requested,
                resolved="cuda:0",
                use_cpu=False,
                hf_device=0,
                cuda_index=0,
            )
        return ResolvedDevice(
            requested=requested,
            resolved="cpu",
            use_cpu=True,
            hf_device=-1,
            cuda_index=None,
        )

    if cuda_match is None:
        raise ValueError(
            f"Unsupported device `{device}`. Use one of: auto, cpu, cuda, cuda:0, cuda:1."
        )

    if torch is None:
        raise RuntimeError(
            f"CUDA device `{requested}` was requested, but `torch` is not installed in this environment."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device `{requested}` was requested, but CUDA is not available."
        )

    raw_index = cuda_match.group("index")
    cuda_index = int(raw_index) if raw_index is not None else 0
    device_count = int(torch.cuda.device_count())
    if cuda_index >= device_count:
        raise RuntimeError(
            f"CUDA device `{requested}` was requested, but only {device_count} CUDA device(s) are available."
        )

    return ResolvedDevice(
        requested=requested,
        resolved=f"cuda:{cuda_index}",
        use_cpu=False,
        hf_device=cuda_index,
        cuda_index=cuda_index,
    )


def resolve_device_argument(
    *,
    device: str,
    use_cpu_alias: bool = False,
    hf_device_alias: int | None = None,
) -> ResolvedDevice:
    requested = (device or "auto").strip().lower() or "auto"
    alias_request: str | None = None

    if use_cpu_alias:
        warnings.warn(
            "`--use-cpu` is deprecated; use `--device cpu` instead.",
            FutureWarning,
            stacklevel=2,
        )
        alias_request = "cpu"

    if hf_device_alias is not None:
        warnings.warn(
            "`--hf-device` is deprecated; use `--device` instead.",
            FutureWarning,
            stacklevel=2,
        )
        alias_request = "cpu" if hf_device_alias < 0 else f"cuda:{hf_device_alias}"

    if alias_request is not None and requested != "auto" and requested != alias_request:
        raise ValueError(
            f"Conflicting device options: `--device {requested}` does not match legacy setting `{alias_request}`."
        )

    return resolve_device_spec(alias_request or requested)
