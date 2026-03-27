CUDA_VISIBLE_DEVICES=1 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --checkpoint checkpoints/phase1_4B/final --n 500
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --no-adapter --n 500


PYTHONPATH=. torchrun --nproc_per_node 2 scripts/train_phase1.py --model Qwen/Qwen3-4B-Thinking-2507 --output checkpoints/phase1_4B/final --epochs 1


PYTHONPATH=. ./venv/bin/python scripts/generate_sft_data.py --backoff-ratio 0.75

# 1. Baseline (no adapter, just the base model)                                                       
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --no-adapter --n 500                                                                                   
                                                                                                          
# 2. Train Phase 1 SFT (1 epoch)                                                                        
PYTHONPATH=. torchrun --nproc_per_node 2 scripts/train_phase1.py --model Qwen/Qwen3-4B-Thinking-2507 --output checkpoints/phase1_4B/final --epochs 1                                                         
                                                                                                        
# 3. Eval SFT'd model                                                                                   
CUDA_VISIBLE_DEVICES=0 ./venv/bin/python -m scripts.eval_phase1 --base-model Qwen/Qwen3-4B-Thinking-2507 --checkpoint checkpoints/phase1_4B/final --n 500 