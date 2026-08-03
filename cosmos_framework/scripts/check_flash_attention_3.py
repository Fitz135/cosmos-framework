# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Fail-fast CUDA preflight for training runs that require Flash Attention 3."""

import os

import torch

FORCED_BACKEND = "flash3"
REQUIRED_COMPUTE_CAPABILITY = (9, 0)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    """Verify that the active GPU environment can execute FA3 forward/backward."""
    configured_backends = os.environ.get("I4_ATTN_BACKENDS")
    _require(
        configured_backends == FORCED_BACKEND,
        f"I4_ATTN_BACKENDS must be exactly 'flash3' for this training run; got {configured_backends!r}.",
    )
    _require(torch.cuda.is_available(), "Flash Attention 3 requires a CUDA GPU, but CUDA is unavailable.")

    device = torch.device("cuda", torch.cuda.current_device())
    capability = torch.cuda.get_device_capability(device)
    device_name = torch.cuda.get_device_name(device)
    _require(
        capability == REQUIRED_COMPUTE_CAPABILITY,
        "The forced Flash Attention 3 path requires an SM90 Hopper GPU "
        f"(H100/H200); got {device_name!r} with compute capability {capability}.",
    )

    try:
        import flash_attn_3_nv
    except Exception as error:
        raise RuntimeError(
            "flash-attn-3-nv could not be imported. Install the dependency group "
            "matching the active Torch/CUDA ABI (for example cu128-train)."
        ) from error

    from cosmos_framework.model.attention.backends import choose_backend, get_backend_list
    from cosmos_framework.model.attention.flash3 import FLASH3_SUPPORTED, flash3_attention

    _require(
        FLASH3_SUPPORTED,
        "Cosmos detected flash-attn-3-nv, but its version or CUDA runtime is unsupported.",
    )

    arch_tag = capability[0] * 10 + capability[1]
    filtered_backends = get_backend_list(arch_tag)
    _require(
        filtered_backends == [FORCED_BACKEND],
        f"The attention allow-list did not resolve exclusively to flash3; got {filtered_backends!r}.",
    )

    tensor_shape = torch.Size((1, 128, 8, 128))
    selected_backend = choose_backend(
        query_shape=tensor_shape,
        key_shape=tensor_shape,
        value_shape=tensor_shape,
        dtype=torch.bfloat16,
        device=device,
        requires_grad=True,
        is_causal=False,
        causal_type=None,
        is_varlen=False,
        deterministic=False,
        raise_error=True,
    )
    _require(
        selected_backend == FORCED_BACKEND,
        f"Cosmos selected {selected_backend!r} instead of forced backend {FORCED_BACKEND!r}.",
    )

    query = torch.randn(tensor_shape, dtype=torch.bfloat16, device=device, requires_grad=True)
    key = torch.randn(tensor_shape, dtype=torch.bfloat16, device=device, requires_grad=True)
    value = torch.randn(tensor_shape, dtype=torch.bfloat16, device=device, requires_grad=True)
    output = flash3_attention(query, key, value)
    output.float().square().mean().backward()
    torch.cuda.synchronize(device)

    gradients = (query.grad, key.grad, value.grad)
    _require(all(gradient is not None for gradient in gradients), "Flash3 backward did not produce all gradients.")
    _require(
        bool(torch.isfinite(output).all())
        and all(bool(torch.isfinite(gradient).all()) for gradient in gradients if gradient is not None),
        "Flash3 forward/backward produced a non-finite value.",
    )

    print(
        "Flash Attention 3 preflight passed: "
        f"backend={selected_backend}, package={flash_attn_3_nv.__version__}, "
        f"torch={torch.__version__}, cuda={torch.version.cuda}, device={device_name}, sm={arch_tag}."
    )


if __name__ == "__main__":
    main()
