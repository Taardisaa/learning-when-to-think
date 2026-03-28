CUDA_VISIBLE_DEVICES=1 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --checkpoint checkpoints/phase1_4B/final --n 500
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --no-adapter --n 500


PYTHONPATH=. torchrun --nproc_per_node 2 scripts/train_phase1.py --model Qwen/Qwen3-4B-Thinking-2507 --output checkpoints/phase1_4B/final --epochs 1



# Thinking enabled.
PYTHONPATH=. ./venv/bin/python -m scripts.eval_math500 --model Qwen/Qwen3-1.7B --n 500 --tp 2

PYTHONPATH=. ./venv/bin/python -m scripts.eval_math500 --model Qwen/Qwen3-1.7B-Base --n 500 --tp 2

# 
PYTHONPATH=. ./venv/bin/python scripts/generate_sft_real_backoff.py --model Qwen/Qwen3-1.7B --dataset math --num-samples 8 --tp 2

# 
PYTHONPATH=. ./venv/bin/python scripts/generate_sft_real_backoff.py --rollouts data/rollouts_grouped_math_Qwen3-1.7B.jsonl --output data/sft_real_backoff_math_train.jsonl

# Stale!!!


PYTHONPATH=. ./venv/bin/python scripts/generate_sft_data.py --backoff-ratio 0.75

# Baseline (max token is 8192)
cd baselines && python eval_gsm8k.py --model Qwen/Qwen3-1.7B --max-samples 500

# Still Baseline. Should be the same as above. 
PYTHONPATH=. ./venv/bin/python -m scripts.eval_vllm_backoff --base-model Qwen/Qwen3-1.7B --n 500

PYTHONPATH=. ./venv/bin/python -m scripts.train_phase1 --model Qwen/Qwen3-1.7B --data data/sft_real_backoff_train.jsonl --output checkpoints/phase1_1.7B_real_backoff/final --epochs 10 --lr 2e-5 --batch-size 1 --grad-accum 4 --max-seq-len 4096

PYTHONPATH=. ./venv/bin/python -m scripts.eval_vllm_backoff --base-model Qwen/Qwen3-1.7B --checkpoint checkpoints/phase1_1.7B_real_backoff/final --n 500

### The following commands are stale.

# Train Phase 1 SFT (1 epoch)
PYTHONPATH=. torchrun --nproc_per_node 2 scripts/train_phase1.py --model Qwen/Qwen3-1.7B --output checkpoints/phase1_qwen3_1.7B/final --epochs 1

# Eval SFT'd model (max token is 8192)
PYTHONPATH=. ./venv/bin/python -m scripts.eval_vllm_backoff --base-model Qwen/Qwen3-1.7B --checkpoint  checkpoints/phase1_qwen3_1.7B/final --n 500


PYTHONPATH=. ./venv/bin/python scripts/generate_sft_from_rollouts.py --model Qwen/Qwen3-1.7B --tp 2

PYTHONPATH=. ./venv/bin/python scripts/generate_sft_real_backoff.py --model Qwen/Qwen3-1.7B --subset 200 --num-samples 8


# 1. Baseline (no adapter, just the base model)                                                       
# CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --no-adapter --n 500                                                                                   
                                                                                                          
# 2. Train Phase 1 SFT (1 epoch)                                                                        
# PYTHONPATH=. torchrun --nproc_per_node 2 scripts/train_phase1.py --model Qwen/Qwen3-4B-Thinking-2507 --output checkpoints/phase1_4B/final --epochs 1

# PYTHONPATH=. torchrun --nproc_per_node 2 scripts.eval_vllm_backoff --model Qwen/Qwen3-1.7B --output checkpoints/phase1_qwen3_1.7B/final --epochs 1
                                                                                                        
# 3. Eval SFT'd model                                                                                   
# CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --checkpoint checkpoints/phase1_4B/final --n 500 