CUDA_VISIBLE_DEVICES=1 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --checkpoint checkpoints/phase1_4B/final --n 500
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --no-adapter --n 500


PYTHONPATH=. torchrun --nproc_per_node 2 scripts/train_phase1.py --model Qwen/Qwen3-4B-Thinking-2507 --output checkpoints/phase1_4B/final --epochs 1


PYTHONPATH=. ./venv/bin/python scripts/generate_sft_data.py --backoff-ratio 0.75

# Baseline (max token is 8192)
cd baselines && python eval_gsm8k.py --model Qwen/Qwen3-1.7B --max-samples 500

# Still Baseline. Should be the same as above. 
PYTHONPATH=. ./venv/bin/python -m scripts.eval_vllm_backoff --base-model Qwen/Qwen3-1.7B --n 500


# Train Phase 1 SFT (1 epoch)
PYTHONPATH=. torchrun --nproc_per_node 2 scripts/train_phase1.py --model Qwen/Qwen3-1.7B --output checkpoints/phase1_qwen3_1.7B/final --epochs 1

# Eval SFT'd model (max token is 8192)
PYTHONPATH=. ./venv/bin/python -m scripts.eval_vllm_backoff --base-model Qwen/Qwen3-1.7B --checkpoint  checkpoints/phase1_qwen3_1.7B/final --n 500

# 1. Baseline (no adapter, just the base model)                                                       
# CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --no-adapter --n 500                                                                                   
                                                                                                          
# 2. Train Phase 1 SFT (1 epoch)                                                                        
# PYTHONPATH=. torchrun --nproc_per_node 2 scripts/train_phase1.py --model Qwen/Qwen3-4B-Thinking-2507 --output checkpoints/phase1_4B/final --epochs 1

# PYTHONPATH=. torchrun --nproc_per_node 2 scripts.eval_vllm_backoff --model Qwen/Qwen3-1.7B --output checkpoints/phase1_qwen3_1.7B/final --epochs 1
                                                                                                        
# 3. Eval SFT'd model                                                                                   
# CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --checkpoint checkpoints/phase1_4B/final --n 500 