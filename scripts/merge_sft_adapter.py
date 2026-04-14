"""Merge SFT LoRA adapter + new token embeddings into base model.

vLLM's LoRA loader doesn't handle the full `lm_head` / `embed_tokens`
weights we save to preserve trained new-token embeddings. This script
merges everything into a standalone full-weight model that vLLM can
load normally.

Usage:
    python -m scripts.merge_sft_adapter \
        --adapter checkpoints/sft_pivot/final \
        --output  checkpoints/sft_pivot/merged \
        --gpus 1
"""

import argparse
import os
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="Merge SFT LoRA + embeddings into full model")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--output", required=True, help="Output directory for merged model")
    parser.add_argument("--base-model", default=None,
                        help="Base model name (default: read from train_config.yaml)")
    parser.add_argument("--gpus", default=None)
    args = parser.parse_args()

    if args.gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    # Lazy imports after CUDA_VISIBLE_DEVICES
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import yaml

    # Figure out base model
    base_model_name = args.base_model
    if base_model_name is None:
        config_path = Path(args.adapter) / "train_config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            base_model_name = cfg["model"]
            print(f"Base model from train_config.yaml: {base_model_name}")
        else:
            raise ValueError("Specify --base-model or provide train_config.yaml in adapter dir")

    # Load tokenizer from adapter (has the new tokens)
    print(f"\nLoading tokenizer from {args.adapter}...")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter)
    print(f"Vocab size: {len(tokenizer)}")

    # Load base model and resize to match adapter's tokenizer
    print(f"\nLoading base model: {base_model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16
    )
    # Resize embeddings to match tokenizer (adds rows for new tokens)
    model.resize_token_embeddings(len(tokenizer))

    # Attach LoRA adapter (this will also load saved embed_tokens/lm_head)
    print(f"\nAttaching adapter from {args.adapter}...")
    model = PeftModel.from_pretrained(model, args.adapter)

    # Merge LoRA weights into base
    print("Merging LoRA into base model...")
    model = model.merge_and_unload()

    # Save the full model
    print(f"\nSaving merged model to {args.output}...")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    # Save generation config if present
    if hasattr(model, "generation_config"):
        model.generation_config.save_pretrained(args.output)

    print(f"\nDone. vLLM can now load from: {args.output}")


if __name__ == "__main__":
    main()
