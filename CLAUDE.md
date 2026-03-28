

# Important Notes

## Dataset generation

when asked to refine or rewrite the dataset, you will:
- avoid using a direct script-based approach to modify the dataset. This will add false positive/false negative cases, which will pollute the dataset. Therefore this is EXTREMELY DISCOURAGED.
- You are encouraged to directly change the entry, by a smart system, e.g., a smarter LLM. This usually refers to yourself, but you may wanna assign subagents to audit each entry, without flooding your own context window. Note that you will assign only ONE entry per subagent.
- For the injected `<backoff_1>`, `<backoff_2>`, `<backoff_3>` tokens, you should apply all of the three tokens into the dataset. 每一个entry可以根据需要随机注入1-3个backoff tokens，他们follow如下format：[N chunks of wrong reasonings, chunks are separated by their natural semantic boundaries] -> <backoff_N> -> [1 chunk of self-corrective tokens with new/corrective directive] -> ...
- `<backoff_1>`, `<backoff_2>`, `<backoff_3>` tokens 应该在perturbed entries中遵循如下分布：60%, 20%, 10%。
- backoff tokens should not have overlaps. To avoid this, try to keep them separated from multiple semantic boundaries within an entry. 
- The perturbated entry should be LENGTH PRESERVING. The original CoT chain should be preserved to prevent from breaking the model's original reasoning policy. Do not attempt to make any truncations, shortenings. Keep it verbose.
- The perturbated entry should be SEMANTIC PRESERVING. The entry should have a clear reasoning chain that finally leads to the correct answer.