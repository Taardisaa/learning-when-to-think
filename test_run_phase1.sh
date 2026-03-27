PYTHONPATH=. torchrun --nproc_per_node 2 scripts/train_phase1.py --epochs 1 --output checkpoints/phase1/test --max-seq-len 512

  # Phase 1 SFT on 4B (2 GPUs, ~4-5 hours)                                                                                                                            
  PYTHONPATH=. torchrun --nproc_per_node 2 scripts/train_phase1.py --model Qwen/Qwen3.5-4B --output checkpoints/phase1_4B/final --epochs 3                            
                                                                                                                                                                      
  After that finishes, Phase 2 GRPO:                                                                                                                                  
                                                                                                                                                                      
  # Phase 2 GRPO (single GPU, ref model on same GPU)        
  python -m scripts.train_phase2 --base-model Qwen/Qwen3.5-4B --phase1 checkpoints/phase1_4B/final --steps 500 --num-rollouts 16 --batch-size 2 --output              
  checkpoints/phase2_4B                                                                                                                                               
                                                                                                                                                                      
  The 4B model might need --max-seq-len 1024 if memory gets tight during SFT with 2048 — but our data maxes at 554 tokens so 1024 is safe.    

PYTHONPATH=. torchrun --nproc_per_node 2 scripts/train_phase1.py --model Qwen/Qwen3.5-4B --output checkpoints/phase1_4B/final_fixed --epochs 3 
python -m scripts.eval_phase1 --checkpoint checkpoints/phase1_4B/final_fixed --n 10