Interesting Cases

  1. Distance — count edges on the path between any two words. "The" to "mat" = 3 edges      
  (The→cat→sat→on→mat... wait, actually The→cat→sat→on, on→mat = 4). The probe learns to     
  predict these. 

  

1. Class imbalance 处理不够 — 当前用 pos_weight 加权，但 20:1 的不平衡可能需要更aggressive 的处理（undersample negatives, focal loss）
Boundary 定义本身可能太 noisy — LLM 标注的 boundary 和 token 级别对齐可能有偏移，一个 char offset 差几位就标到错的 token 上了
Layer 6 最好而不是后面的层 — 这和 Zhang/Afzal 的发现（mid-late layer 最好）不一致，可能说明 boundary 信息更像句法（early-mid）而不是推理 correctness（mid-late）
没跑 displacement — 结果里全是 raw，displacement 没出来，可能是因为层不连续（6,12,18,24 间隔太大，compute_displacement 只对连续层生效