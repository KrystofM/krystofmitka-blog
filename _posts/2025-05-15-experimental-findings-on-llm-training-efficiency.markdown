---
layout: post
title: "Experimental Findings on LLM Training Efficiency"
date: 2025-05-15 10:00:00 +0200
excerpt: "Practical experiments to speed up LLM training: ISL, VSL+DD, cross‑vocab embeddings (~30% speedup), multi‑token, MoE, with math on EOT retention and transfer."
categories: machine-learning research
author:
  name: "Krystof Mitka"
  link: "https://krystofmitka.com"
info_banner: "This work was conducted as part of the <a href='https://github.com/BottleCapAI/NoCap-Test' target='_blank'>BottleCapAI NoCap-Test</a>, an open challenge focused on making large language models radically more efficient through algorithmic improvements."
---

## Introduction

This work explores various algorithmic techniques to improve LLM training efficiency. The goal is to train a GPT-2 style model to reach a validation loss of $$\mathcal{L}_{val} \leq 3.3821$$ on the FineWeb dataset faster than the baseline.

**Experimental Setup:**
- **Model**: GPT-2 architecture, 124M parameters (12 layers, 768 hidden dimensions, 12 attention heads)
- **Hardware**: Single NVIDIA L40 GPU
- **Dataset**: FineWeb (pre-tokenized with GPT-2 tokenizer), 5B token training budget
- **Optimizer**: Muon optimizer
- **Baseline**: 5.4 hours to reach target validation loss of 3.3821
- **Vocabulary**: 50,257 tokens (GPT-2 tokenizer)

The following sections document various experimental approaches, including curriculum learning strategies, embedding initialization techniques, architectural modifications, and their results compared to the baseline.


## Incremental Sequence Length Training

I started by training the model incrementally on sequences of lengths 128, 256, 512, and 1024. The increments were applied after 128 or 256 steps, and the effective batch sizes were not initially scaled to match each other in total tokens per update. 

The training loss dropped significantly faster at the beginning with lower sequence lengths. However, this advantage wore off toward the end of training and was eventually surpassed by standard 1024-token training. This performance drop could be attributed to either the unequal batch sizes resulting in faster weight updates early on, or to gradient updates from shorter sequences becoming less relevant during later training, an exhibit of model unlearning. 

After properly scaling the effective batch size, the model's training loss matched the stand-alone 1024 sequence length training [[1]](#ref-1). I suspect better results could be achieved with larger models and longer context windows.

## Variable Sequence Length Training

Building on the ISL approach, I implemented VSL with Equal, Grow-P2, and Grow-Linear curricula [[2]](#ref-2) and properly scaled batch sizes [[3]](#ref-3). Unlike ISL's deterministic progression, VSL randomly samples sequence lengths during each training step according to probabilities set by the curriculum.

Unfortunately, training became more unstable at certain points as gradients started to explode, even with gradient clipping enabled. While using lower learning rates than the baseline improved stability, this somewhat defeated the purpose of achieving faster training. Additionally, there was a noticeable jump in validation loss at the end of each cycle when the training shifted back to sampling shorter sequence lengths.

## Data Decomposition into Buckets

To support VSL training more effectively, I implemented Data Decomposition [[2]](#ref-2) [[3]](#ref-3) of the RefinedWeb dataset, which splits documents into buckets of individual sequence lengths: 128, 256, 512, 1024, etc. Surprisingly, this resulted in a bump in validation loss. It performed slightly slower than baseline even without VSL active.

I realized something interesting here: we are effectively not teaching the model where documents end, which could be problematic. Since DD uses binary decomposition of documents, and all chunks smaller than 128 tokens are discarded, the EOT token (which appears at the end of each document) is only retained if it ends up in a chunk of size ≥ 128. 

In binary decomposition, a document of length $$L$$ is split according to its binary representation. The EOT token ends up in the smallest (rightmost) chunk. For the EOT to be retained, this smallest chunk must be ≥ 128, which only occurs when $$L \equiv 0 \pmod{128}$$. For typical document length distributions (e.g., bounded between 100 and 10k tokens with mean ~500), $$L \bmod 128$$ is approximately uniformly distributed over $$\{0, 1, \ldots, 127\}$$, giving:

<div class="math-container">
$$
P(\text{EOT retained}) = P(L \equiv 0 \pmod{128}) \approx \frac{1}{128} \approx 0.78\%
$$
</div>

This means approximately 99.2% of training sequences exclude the EOT token. 

One potential solution would be decomposing documents from the end rather than the start, though this could create other problems like insufficient training on document beginnings. In retrospect, using attention masking to handle variable-length sequences within a batch would likely be more appropriate and significantly easier to implement than binary decomposition.

**Communication with the Apple ML Team:** I reached out to the paper's lead author (Hadi Pouransari) with these findings. His response provided helpful guidance on reproducing the paper's results, including insights on the EOT token issue and the importance of data quality. The full correspondence is included in the [Appendix](#appendix).

## Initializing the Embedding Layer Using a Large Pre-Trained Model

This approach proved to be simple yet remarkably effective, though it can be considered cheating. The basic idea is to initialize our model's embeddings using those from a larger, pre-trained model. For higher embedding dimensions, one can either concatenate dimensions or use more sophisticated methods like PCA to extract the most powerful components. The main challenge, however, is dealing with different vocabularies across models. Our GPT-2 model has a vocabulary size of 50K, while more recent models use significantly larger vocabularies with non-compatible tokenizations.

**Cross-Vocab Embedding Transfer:** From my previous research on extracting embedding layers from GPT-3.5/4 [[4]](#ref-4), I developed a method to transfer embeddings between different vocabularies. The approach is straightforward: 

For each token $$t_{target}$$ in the target vocabulary $$\mathcal{V}_{target}$$ (e.g., GPT-2), use the source tokenizer to decompose it into source tokens $$\{s_1, s_2, \ldots, s_k\} \subseteq \mathcal{V}_{source}$$ (e.g., from Qwen or DeepSeek). Assume the source has a larger embedding dimension than the target ($$d_{\text{source}} > d_{\text{target}}$$). Let the embedding matrices be:

<div class="math-container">
$$
W_{\text{source}} \in \mathbb{R}^{\lvert \mathcal{V}_{source} \rvert \times d_{\text{source}}}, \quad
W_{\text{target}} \in \mathbb{R}^{\lvert \mathcal{V}_{target} \rvert \times d_{\text{target}}}
$$
</div>

Using row-slice notation, define the target token embedding with a PCA projection $$P \in \mathbb{R}^{d_{\text{source}} \times d_{\text{target}}}$$ (fitted on rows of $$W_{\text{source}}$$) applied after summation:

<div class="math-container">
$$
W_{\text{target}}[t,:] = \left( \sum_{i=1}^{k} W_{\text{source}}[s_i,:] \right) P
$$
</div>

where $$W_{\text{source}}[s_i,:]$$ is the pre-trained embedding row for the source token $$s_i$$ and $$P$$ reduces the summed source vector from $$d_{\text{source}}$$ to $$d_{\text{target}}$$.

There are obvious limitations to this technique. For example, if the target vocabulary has a token for "hotdog" but the source tokenizer decomposes it into ["hot", "dog"], the summed embedding may not perfectly capture the semantic meaning since the components have little connection to the whole.

**Universal Embedding Geometry:** Interestingly, recent research [[5]](#ref-5) [[6]](#ref-6) suggests this approach might not be cheating after all. These papers hypothesize and demonstrate that all learned embeddings share an underlying geometric structure; essentially, they're all translatable to the same representation space. Tomas Mikolov also pointed me to earlier work on this topic [[25]](#ref-25), which showed similar geometric representations across languages in translation models, providing early evidence for this universal structure.

**Results:** When testing with embeddings extracted and converted from larger models like Qwen or DeepSeek using cross-vocab transfer, the training curve showed a sharp gain at the beginning, achieving an overall training speed increase of approximately 30%.

**Why This Works So Well:** For a small model like GPT2-124M, the tied embedding matrix accounts for a significant fraction of all parameters:

<div class="math-container">
$$
\frac{|\mathcal{V}| \times d_{model}}{N_{total}} = \frac{50257 \times 768}{124M} \approx 0.31
$$
</div>

This 31% parameter share explains the dramatic performance improvement. For larger models (e.g., 1.5B parameters), the embedding matrix represents only ~5% of total parameters, yielding a smaller but still meaningful boost.

## DyT Layers

Not all experiments were successful. I experimented with replacing RMS normalization [[7]](#ref-7) with dynamic tanh layers as proposed in [[8]](#ref-8). However, rather than the reported improvements, I experienced gradient explosion under the initial learning rate, making training unstable. This highlights that techniques don't always transfer across different scales and setups.

## Multi-Token Prediction with Extra Linear Layers

I also implemented multi-token prediction by adding additional linear layers (unembedding layers) at the end of the model [[9]](#ref-9). The approach shares hidden states before mapping to the vocabulary, with auxiliary losses for each prediction head. However, this significantly increases model size. Each additional predicted token requires:

<div class="math-container">
$$
N_{params} = |\mathcal{V}| \times d_{model} = 50257 \times 768 \approx 38M \text{ parameters}
$$
</div>

I observed no speed gains, consistent with the paper's findings [[9]](#ref-9) that relative benefits of multi-token prediction diminish for small models. This makes sense: the extra layers represent a substantial fraction of our 124M parameter model. I would expect better results on models of 300M+ parameters where the additional layers are a smaller proportion of the total.

## NoPE Layers & Scalable Softmax

To improve model generalization across variable sequence lengths, I made two architectural modifications. First, I removed Rotary Position Embeddings (RoPE) [[10]](#ref-10) from most layers, only applying it to every 4th layer, an approach known as NoPE (No Position Encoding) layers [[11]](#ref-11) [[12]](#ref-12) [[13]](#ref-13). Specifically, RoPE is applied only at layers $$\ell \in \{4, 8, 12\}$$. This should theoretically help the model adapt better when switching between short and long sequences during curriculum training.

Second, I implemented scalable softmax [[14]](#ref-14), which dynamically adjusts attention scaling based on input vector size. This was intended to further improve generalization when transitioning between different sequence lengths.

## GQA & MQA Attention and MoE Experiments

I explored several alternative attention mechanisms and architectural patterns. First, I implemented both Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) [[15]](#ref-15), which reduce KV-cache size and speed up inference. However, since this benchmark measures training speed rather than inference speed, the reduced parameter count provided no benefits. Moreover, GQA and MQA lack the kernel-level optimizations of PyTorch's FlashAttention-2 [[16]](#ref-16), resulting in actually slower performance.

I also experimented with a mixture-of-experts (MoE) architecture incorporating both shared and specialized experts, following the DeepSeek approach [[17]](#ref-17) [[18]](#ref-18). The model uses top-2 gating, routing each token to the two highest-scoring experts, and a load-balancing loss ensures even distribution across experts. 

While validation loss initially improved faster, the benefits diminished over time and eventually converged to baseline performance. This likely stems from routing overhead, sparse updates to individual experts, and the fact that MoE primarily improves inference efficiency rather than training speed, especially at smaller scales like ours.

<table class="responsive-table">
  <thead>
    <tr>
      <th>Technique</th>
      <th>Result</th>
      <th>Estimated Speed Gain</th>
      <th>Key Failure Mode</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td data-label="Technique">Incremental SL</td>
      <td data-label="Result">Matched baseline after scaling</td>
      <td data-label="Estimated Speed Gain">~0%</td>
      <td data-label="Key Failure Mode">Forgetting later</td>
    </tr>
    <tr>
      <td data-label="Technique">VSL + DD</td>
      <td data-label="Result">Instability / EOT loss issue</td>
      <td data-label="Estimated Speed Gain">Negative</td>
      <td data-label="Key Failure Mode">Gradient spikes</td>
    </tr>
    <tr>
      <td data-label="Technique">Cross-vocab Embeddings</td>
      <td data-label="Result"><strong>Major win</strong></td>
      <td data-label="Estimated Speed Gain"><strong>~30% faster</strong></td>
      <td data-label="Key Failure Mode">Requires tokenizer bridging</td>
    </tr>
    <tr>
      <td data-label="Technique">Multi-token Heads</td>
      <td data-label="Result">No improvement</td>
      <td data-label="Estimated Speed Gain">0%</td>
      <td data-label="Key Failure Mode">Over-parameterization</td>
    </tr>
    <tr>
      <td data-label="Technique">MoE</td>
      <td data-label="Result">Neutral</td>
      <td data-label="Estimated Speed Gain">0–5%</td>
      <td data-label="Key Failure Mode">Routing overhead, scale too small</td>
    </tr>
  </tbody>
</table>

## BONUS: GRPO Training for Optimized CUDA Kernels

As a tangent unrelated to the NoCap test, I also worked on training models to generate optimized CUDA kernels. For several weeks, I had access to a cluster of 8×B200 GPUs where I trained a Qwen R1 Distilled 1B model [[19]](#ref-19) using Group Reward Policy Optimization [[20]](#ref-20). The task: given PyTorch code, produce optimized CUDA kernels using the KernelBench dataset [[21]](#ref-21). The reward function was defined as:

<div class="math-container">
$$
R(\text{kernel}) = \begin{cases}
\frac{t_{\text{PyTorch}}}{t_{\text{CUDA}}} & \text{if compilation succeeds} \\
0 & \text{otherwise}
\end{cases}
$$
</div>

where $$t_{\text{PyTorch}}$$ and $$t_{\text{CUDA}}$$ represent execution times.

**Training setup:** I used an asymmetric setup with 1 GPU running inference via vLLM [[22]](#ref-22) and the remaining 7 GPUs for training. ZeRO Stage 3 [[23]](#ref-23) partitioned optimizer states, gradients, and model parameters across all GPUs.

**Reward hacking:** An interesting challenge emerged during training, the model discovered it could maximize rewards by simply copying the input PyTorch code verbatim, which compiles successfully but provides no optimization. This is a classic reward hacking problem.

**Related work:** Interestingly, a few weeks after my experiments, Cognition published very similar work on *Multi-Turn RL for Generating CUDA Kernels* [[24]](#ref-24). They used the same base model and encountered identical reward hacking issues. Their extension allowed multi-turn compilation where the model observes intermediate results.


## References

<span id="ref-1">[1]</span> Li, C., Zhang, M., & He, Y. (2022). The Stability-Efficiency Dilemma: Investigating Sequence Length Warmup for Training GPT Models. [arXiv:2108.06084](https://arxiv.org/abs/2108.06084)

<span id="ref-2">[2]</span> Pouransari, H., Li, C. L., Chang, J. H. R., Vasu, P. K. A., Koc, C., Shankar, V., & Tuzel, O. (2025). Dataset Decomposition: Faster LLM Training with Variable Sequence Length Curriculum. [arXiv:2405.13226](https://arxiv.org/abs/2405.13226)

<span id="ref-3">[3]</span> Yang, Q., Peng, Q., Liu, H., Liu, K., Qin, B., & Liu, T. (2025). Beyond Fixed Length: Bucket Pre-training is All You Need. [arXiv:2407.07495](https://arxiv.org/abs/2407.07495)

<span id="ref-4">[4]</span> Mitka, K. (2024). Stealing a Part of Production LM: Improving the Algorithm. [Blog post](https://blog.krystofmitka.com/2024/10/31/stealing-part-of-production-model.html)

<span id="ref-5">[5]</span> Jha, R., Zhang, C., Shmatikov, V., & Morris, J. X. (2025). Harnessing the Universal Geometry of Embeddings. [arXiv:2505.12540](https://arxiv.org/abs/2505.12540)

<span id="ref-6">[6]</span> Huh, M., Cheung, B., Wang, T., & Isola, P. (2024). The Platonic Representation Hypothesis. [arXiv:2405.07987](https://arxiv.org/abs/2405.07987)

<span id="ref-7">[7]</span> Zhang, B., & Sennrich, R. (2019). Root Mean Square Layer Normalization. [arXiv:1910.07467](https://arxiv.org/abs/1910.07467)

<span id="ref-8">[8]</span> Zhu, J., Chen, X., He, K., LeCun, Y., & Liu, Z. (2025). Transformers without Normalization. [arXiv:2503.10622](https://arxiv.org/abs/2503.10622)

<span id="ref-9">[9]</span> Gloeckle, F., Idrissi, B. Y., Rozière, B., Lopez-Paz, D., & Synnaeve, G. (2024). Better & Faster Large Language Models via Multi-token Prediction. [arXiv:2404.19737](https://arxiv.org/abs/2404.19737)

<span id="ref-10">[10]</span> Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2023). RoFormer: Enhanced Transformer with Rotary Position Embedding. [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)

<span id="ref-11">[11]</span> Kazemnejad, A., Padhi, I., Ramamurthy, K. N., Das, P., & Reddy, S. (2023). The Impact of Positional Encoding on Length Generalization in Transformers. [arXiv:2305.19466](https://arxiv.org/abs/2305.19466)

<span id="ref-12">[12]</span> Meta AI (2024). The Llama 3 Herd of Models. [arXiv:2407.21783](https://arxiv.org/abs/2407.21783)

<span id="ref-13">[13]</span> Peng, B., Quesnelle, J., Fan, H., & Shippole, E. (2023). YaRN: Efficient Context Window Extension of Large Language Models. [arXiv:2309.00071](https://arxiv.org/abs/2309.00071)

<span id="ref-14">[14]</span> Nakanishi, K. M. (2025). Scalable-Softmax Is Superior for Attention. [arXiv:2501.19399](https://arxiv.org/abs/2501.19399)

<span id="ref-15">[15]</span> Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebrón, F., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. [arXiv:2305.13245](https://arxiv.org/abs/2305.13245)

<span id="ref-16">[16]</span> Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

<span id="ref-17">[17]</span> Dai, D., et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models. [arXiv:2401.06066](https://arxiv.org/abs/2401.06066)

<span id="ref-18">[18]</span> DeepSeek-AI (2025). DeepSeek-V3 Technical Report. [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)

<span id="ref-19">[19]</span> DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)

<span id="ref-20">[20]</span> Shao, Z., et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)

<span id="ref-21">[21]</span> Ouyang, A., et al. (2025). KernelBench: Can LLMs Write Efficient GPU Kernels? [arXiv:2502.10517](https://arxiv.org/abs/2502.10517)

<span id="ref-22">[22]</span> Kwon, W., et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)

<span id="ref-23">[23]</span> Rajbhandari, S., Rasley, J., Ruwase, O., & He, Y. (2020). ZeRO: Memory Optimizations Toward Training Trillion Parameter Models. [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)

<span id="ref-24">[24]</span> Baronio, C., Marsella, P., Pan, B., Guo, S., & Alberti, S. (2025). Kevin: Multi-Turn RL for Generating CUDA Kernels. [arXiv:2507.11948](https://arxiv.org/abs/2507.11948)

<span id="ref-25">[25]</span> Mikolov, T., Le, Q. V., & Sutskever, I. (2013). Exploiting Similarities among Languages for Machine Translation. [arXiv:1309.4168](https://arxiv.org/abs/1309.4168)


<a id="appendix"></a>
## Appendix: Correspondence with Hadi

```
Hi Krystof,

See my comments below:

On May 23, 2025, at 6:57AM, Krystof Mitka <krystof.mitka@student.uva.nl> wrote:

Hi Hadi,

I wanted to reach out because I have spent the past few days trying to replicate 
the results of your paper "Dataset Decomposition: Faster LLM Training with Variable 
Sequence Length Curriculum" on the KernelJordan/ModdedNanoGPT repo. The model I am 
using is relatively small - 150M parameters, with 1024 sequence length maximum - 
for VSL I am using sequence lengths 128, 256, 512 and 1024; Grow-P2; 8 cycles.

The official implementation is based on the open_lm repo, and can be found here 
as a patch. There is also a 160m model in open_lm. I recommend first trying the 
official implementation, and then migrating it to another repo.

VSL

With VSL implemented w Grow-P2 curriculum and scaled batch sized, the training 
becomes unstable at a certain point as the gradient start exploding - even when 
accounted for gradient clipping. The training is stable with a lower learning rate 
than the baseline, but that slightly beats the purpose of the potential speed up. 
There is also a noticeable jump in validation loss at the end of the cycles as the 
training shifts to lower sequence lengths again.

Some variation in the training loss is expected, as loss is a function of sequence 
length. I suggest going with the default parameter in the released repo, and 
starting from that. You can also start with no curriculum and equally distributed 
mixture. We did not observe any instability. One major source of loss instability 
is low-quality data. Make sure to use well-filtered data. For example, DCLM could 
be a good one.

DD

For DD I have noticed just a bump in the overall validation loss - slightly slower 
than baseline even with just VSL active. I had some trouble with the EOT token. 
As DD uses binary decomposition of the document - most sequences will not have the 
EOT token included ( just in 50% of the cases it would be decomposed to a sequence 
length of 1 which gets discarded, for sequences up to 128 that chance becomes even 
higher ~98%, essentially guaranteeing the EOT token will not be included in training 
much ). One solution could be that the documents are decomposed from the end of the 
document and not start, which I have yet to test - but that could run into other 
problems such as not being able to train on the beginnings of the documents. For 
validation I am also using the concat-and-chunk-document dataset and not the bucketed 
dataset of length 1024, which is something I could experiment with further.

Not sure why your validation gets worse. We showed in the paper that distribution 
match (from sequence length perspective) matters. Depending on what you use for 
validation, you may want to modify your training mixture (how many samples from 
each bucket).

Most core pre-training evaluations do not depend on EOT (e.g., multiple choice 
questions compare perplexity of each solution, etc.). Often, generation termination 
is learned during post-training (SFT, RLHF, etc.). But if you like to have more EOT 
during your pretraining, you can go with what you suggest (truncate from the end of 
sequence) or use padding.

Let me know if there is something I might have missed in my implementation; I can 
also provide the code and logs from the training runs. It could be that the model 
is too small to make a noticeable difference, however, so far, the claimed speed 
ups in training times and token efficiency do not seem to hold with my setup. I 
might make a short blog post describing the results and methodology used.

I don't think small model is a problem. For the paper we tested 160m model with 
sequences up to 8192 from the DCLM dataset, and observed ~15% speed up. See Table 3 
in the paper.

If you do not observe speed up you can:

1) Go to a higher sequence length (e.g., 8192, or more).

2) Check if all GPUs for every optimization step get the same sequence length. If 
this is not happening, you won't get a speed up, since you will be bottlenecked by 
the longest sequence length in the batch.

Good luck with your project!

Hadi
```
