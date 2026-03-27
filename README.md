# learning-when-to-think

- enable_thinking: true
- max token: 2048
- temperature: 0.0 (greedy)
- repetition penalty: 1.3
- model: Qwen3.5-4B

Training: 3 epochs

LoRA parameters:

- epochs: 3
- learning rate: 2e-5
- batch size: 1
- grad accum: 4
- max seq: 2048
- lora-r: 16
- lora-alpha: 32
- dtype: bfloat16
- target modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
- lora dropout: 0.05
- bias: None
- task type: "CASUAL_LM"
- lr scheduler: "cosine"
- warmup ratio: 0.03

## Forced Masking (Deprecated)

Trained using LoRA SFT for warm up.

`python -m scripts.eval_phase1 --base-model Qwen/Qwen3.5-4B --checkpoint checkpoints/phase1_4B/final_fixed --n 500`

SUMMARY (500 problems)

Accuracy:     99/500 (20%)
- `<continue>`:   1180 total (2.4/problem)
- `<backoff>`:    207 total (0.4/problem)
- `Terminated`:   498/500

## Base Model 

`python -m scripts.eval_phase1 --base-model Qwen/Qwen3.5-4B --no-adapter --n 500`

