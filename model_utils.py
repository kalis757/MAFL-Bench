"""Shared model and tokenizer loading."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(config: dict):
    """Load a HuggingFace model and tokenizer from config.

    Config keys read (all under config["model"]):
      name_or_path  — HF Hub ID or local path
      quantization  — "none" | "4bit" | "8bit"  (default: "none")
                      "4bit" → BitsAndBytes NF4 quantization
                      "8bit" → BitsAndBytes 8-bit quantization
      torch_dtype   — "float16" | "bfloat16" | "float32"  (default: "float16")
      device_map    — passed to from_pretrained (default: "auto")
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_cfg = config["model"]
    model_path = model_cfg["name_or_path"]
    quantization = model_cfg.get("quantization", "none")
    device_map = model_cfg.get("device_map", "auto")
    torch_dtype_str = model_cfg.get("torch_dtype", "float16")

    torch_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(torch_dtype_str, torch.float16)

    logger.info("Loading tokenizer from %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(
        "Loading model (quantization=%s, dtype=%s, device_map=%s)",
        quantization, torch_dtype_str, device_map,
    )

    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map=device_map
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map=device_map
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map=device_map
        )

    model.eval()
    logger.info("Model loaded.")
    return model, tokenizer
