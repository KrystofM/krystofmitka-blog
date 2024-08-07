---
layout: post
title: "Stealing Part of Production Language Model"
date: 2024-07-26 17:48:50 +0200
excerpt: summary here
---

## General Algorithm Formulation

We consider a typical transformer architecture for a large language model. Let $\mathcal{P(X)}$ represent the space of all probability distributions over the vocabulary $\mathcal{X}$. Transformer models take $N$ tokens as input from a vocabulary $\mathcal{X}$ of size $v$ and output a probability distribution for the next token $\mathbf{q} \in \mathcal{P(X)}$, i.e., models of the form $f: \mathcal{X}^N \mapsto \mathcal{P(X)}$. The probability distribution, conditioned on the previous $N$ tokens in the sequence, is computed by applying a softmax $\mathbb{R}^v \mapsto \mathcal{P(X)}$ to the full logit distribution vector $\mathbb{R}^{v}$ of the last token, $\mathbf{z}$.

Logits are the raw, unnormalized scores output by a neural network's final layer before applying the softmax function. The softmax function is defined as:

$$
\text{softmax}(\mathbf{z}) = \left[ \frac{e^{z_1}}{\sum_{i=1}^v e^{z_i}}, \ldots, \frac{e^{z_v}}{\sum_{i=1}^v e^{z_i}} \right]
$$

We assume access to the model API provided by the owners of the model, such as OpenAI or Google, and we explore the class of attacks based on the following two assumptions.

**Access to Bias Map.** We assume that it is possible to set a bias term $b_i$ for any token $\mathcal{X}_i$ in the vocabulary $\mathcal{X}$, where $b_i$ can range from $-B$ to $B$. The number of tokens that can have a bias term set is limited to a constant $N_b \leq v$. We use vector notation $\mathbf{b} \in \mathbb{R}^v$, meaning the length of the bias vector is $v$. In cases where $N_b < v$, for all tokens that are not biased, we assume $b_i = 0$. The bias vector $\mathbf{b}$ is added to the logit vector $\mathbf{z}$ before applying the softmax function.

**Access to Top Token.** We constrain the information received from the API to only the token with the highest probability after applying the softmax function.

Define prompt $p$ as a unique sequence of $N$ tokens and let $g: \mathcal{X}^N \rightarrow \mathbb{R}^v$ be a function that outputs the logit vector $\mathbf{z}$ for the next token in the sequence. The Oracle $\mathcal{O}$ (API) is a black-box function that, given a prompt $p$ and a bias vector $\mathbf{b}$, returns the token with the highest probability.

$$
\mathcal{O}(p, b) \gets \text{ArgMax} \left(\text{softmax} \left( g(p) + \mathbf{b} \right) \right)
$$

Additionally, we assume that $\text{ArgMax}(\mathbf{v})$ returns the index of the coordinate with the largest value in some vector $\mathbf{v} \in \mathbb{R}^v$. In case two or more values are equal and the maximum, the $\text{ArgMax}$ function will randomly select one of the indices corresponding to the maximum values. We refer to the indices of the vectors and tokens interchangeably.

**Definition 1:** We say that a token is **sampled** if it is the token returned by the oracle's ArgMax function.

Note that we can ensure the token with the highest logit value, which corresponds to the highest probability token, is consistently sampled by setting the temperature parameter to 0 or by configuring the top-k parameter to 1.

### Logit normalization

To facilitate the implementation of the algorithm and ensure consistent scaling of the logit values, we will normalize the logit vectors to the interval $[0, 1]$. This normalization helps in simplifying the bias adjustments and maintaining a uniform scale for comparison across different logits.

Given a logit vector $\mathbf{z}$, we normalize it using the following procedure. Calculate the minimum and maximum values of the logit vector $\mathbf{z}$:

$$
z_{\delta} = \min(\mathbf{z}), \quad z_{\Delta} = \max(\mathbf{z})
$$

Normalize each component $z_i$ of the logit vector $\mathbf{z}$ to the interval $[0, 1]$ using the formula:

$$
\hat{z}_i = \frac{z_i - z_{\delta}}{z_{\Delta} - z_{\delta}}
$$

where $\hat{z}_i$ is the normalized logit value.

In a realistic setting, we can assume knowledge of the width of the interval on which logits lie. Observations from API providers suggest that it suffices to take $B$ as the width of this interval. Furthermore, to facilitate shifting the interval and simplify notation, we assume knowledge of the maximum logit value $z_\Delta$.[^1]

Henceforth, we assume all logit vectors $\mathbf{z}$ are normalized to the interval $[0, 1]$.

[^1]: Note that OpenAI provides access to the log probabilities of the top-5 most likely tokens, which are not affected by the bias term after the introduction of the attacks. Consequently, we can infer the initial top-5 token logits, allowing us to determine the maximum value.

### Algorithm Framework

We will consider algorithms that modify the bias of multiple tokens simultaneously. Specifically, at each step, we will adjust the biases of all $N_b$ tokens. In scenarios where $N_b < v$, it is necessary to select subsets of tokens and repeat the procedure $\frac{v}{N_b}$ times. Prior to initiating the algorithm, we perform an initial query to the API without applying any bias map to identify the top token and set its index to 0 i.e. top token is $z_0$.
We now introduce a general algorithm framework that is used to iteratively narrow down the possible values of $\mathbf{z}$, using lower bound vector $\mathbf{l} \in \mathbb{R}^{N_b}$ and upper bound vector $\mathbf{h} \in \mathbb{R}^{N_b}$, and give a concrete description of its implementation. The goal of all instances of this algorithm discussed in the following sections will be to come up with the function $\mathbf{f}(\mathbf{l}, \mathbf{h}, r)$, which sets all $N_b$ bias terms at each step of the algorithm. The following algorithm framework is a vectorized and generalized version of the algorithm discussed in \cite{carlini_stealing_2024}.

<div class="algorithm">
<strong>Algorithm 1:</strong> Learning logit differences with multi-token calls (Vectorized)

<ol>
<li>$\mathbf{l} \gets \mathbf{0}, \mathbf{h} \gets \mathbf{1}$</li>
<li>$\mathcal{C} = \{\mathbf{x} : z_0 - z_i \leq 1, \forall i \}$</li>
<li>$r \gets 1$</li>
<li><strong>while</strong> stop_condition($\mathbf{l}, \mathbf{h}, r$) is false <strong>do</strong></li>
<li style="text-indent: 1em;">$\mathbf{b} \gets \mathbf{f}(\mathbf{l}, \mathbf{h}, r)$</li>
<li style="text-indent: 1em;">$k \gets \mathcal{O}(p, \mathbf{b})$</li>
<li style="text-indent: 1em;">$\mathcal{C} \gets \mathcal{C} \cap \{\mathbf{x} : z_k + b_k \geq z_j + b_j, \forall j \neq k\}$</li>
<li style="text-indent: 1em;">$\mathbf{l} \gets \mathbf{x}_{\text{min}}$</li>
<li style="text-indent: 1em;">$\mathbf{h} \gets \mathbf{x}_{\text{max}}$</li>
<li style="text-indent: 1em;">$r \gets r + 1$</li>
<li><strong>end while</strong></li>
<li><strong>return</strong> $\mathbf{l}, \mathbf{h}$</li>
</ol>
</div>

At each step of Algorithm 1, we provide a specific bias vector $\mathbf{b} = \mathbf{f}(\mathbf{l}, \mathbf{h}, r)$ and sample a top token from the oracle, denoted as $k$. By the definition of $\mathcal{O}$, this results in $N_b - 1$ inequalities of the form $z_k + b_k \geq z_j + b_j$ for all $j \neq k$. We can rewrite these inequalities as $-z_k + z_j \leq b_k - b_j$. Therefore, at step $n$, we can represent the coefficients of logits as $A_n \in \mathbb{R}^{N_b - 1 \times N_b}$ and the resulting bound as $\mathbf{b}_n \in \mathbb{R}^{N_b - 1}$. Aggregating results after $n$ steps, we let:

$$
A = \begin{pmatrix}
A_1 \\
A_2 \\
\vdots \\
A_n
\end{pmatrix}, \quad
b = \begin{pmatrix}
b_1 \\
b_2 \\
\vdots \\
b_n
\end{pmatrix}
$$

So, the polytope $\mathcal{C}$ defined by this system of linear inequalities can be expressed as:

$$\mathcal{C} = \{ \mathbf{x} \in \mathbb{R}^{N_b} \mid A \mathbf{x} \leq \mathbf{b} \}$$

After updating the system for our polytope at each step with new information, we are interested in finding $$\mathbf{x}_{\text{min}}$$ and $$\mathbf{x}_{\text{max}}$$, where $$\mathbf{x}_{\text{min}}$$ is the solution of the polytope where each component of the vector $\mathbf{x}$ is minimal and $$\mathbf{x}_{\text{max}}$$ is the solution where each component is maximal. Formally, this can be defined as:

$$
\mathbf{x}_{\text{min}} = \left( \min_{\mathbf{x} \in \mathcal{C}} x_1, \min_{\mathbf{x} \in \mathcal{C}} x_2, \ldots, \min_{\mathbf{x} \in \mathcal{C}} x_n \right)
$$

and

$$
\mathbf{x}_{\text{max}} = \left( \max_{\mathbf{x} \in \mathcal{C}} x_1, \max_{\mathbf{x} \in \mathcal{C}} x_2, \ldots, \max_{\mathbf{x} \in \mathcal{C}} x_n \right)
$$

We can compute both $$\mathbf{x}_{min}$$ and $$\mathbf{x}_{max}$$ by considering the linear programming problem of finding the shortest path on a weighted graph ([Carlini et al., 2024](#carlini2024)).

Generally, we consider two different stop conditions for the attacker's querying process:

1. **Fixed Budget Condition**: When the attacker has a budget of $T$ requests for each batch of $N_b$ tokens, the querying process stops after $T$ rounds. Formally, this stop condition can be defined as:

   $$
   \text{stop_condition}(\mathbf{l}, \mathbf{h}, r) =
   \begin{cases}
   \text{True} & \text{if } r \geq T, \\
   \text{False} & \text{otherwise}.
   \end{cases}
   $$

2. **Precision Condition**: When the attacker aims to attain a certain precision $\epsilon$ of the result and disregards the budget, the querying process stops when the largest interval so far is less than $\epsilon$. Formally, this stop condition can be defined as:

   $$
   \text{stop_condition}(\mathbf{l}, \mathbf{h}, r) =
   \begin{cases}
   \text{True} & \text{if } \| \mathbf{h} - \mathbf{l} \|_\infty < \epsilon, \\
   \text{False} & \text{otherwise}.
   \end{cases}
   $$

### StartOverN With Uniform Prior

We start by assuming uniform prior over the interval $[l_i, h_i]$ for each logit. A simple approach is to bias logits such that the probability of sampling token 0 (i.e. the maximum token) is $1/n$. This approach, introduced briefly in [Carlini et al., 2024](#carlini2024), has been the state-of-the-art method prior to the work presented in this paper. We formalize this approach and provide a derivation for a solution of the biasing term.

**Theorem 1:** Given independent and identically distributed (i.i.d.) random variables $P_1, P_2, \ldots, P_{n-1}$ from a uniform distribution on an interval between $l_i$ and $h_i$, and constants $b_1, b_2, \ldots, b_{n-1}$, the StartOverN algorithm aims to find $b_i$ such that:

$$
P \left( \max(P_1 + b_1, P_2 + b_2, \ldots, P_{n-1} + b_{n-1}) \leq 1 \right) = \frac{1}{n}
$$

A solution to this problem is given by:

$$
b_i = 1 - l_i - \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (h_i - l_i)
$$

**Proof:**
We need

$$
\begin{align*}
P \left( \max(P_1 + b_1, \ldots, P_{n-1} + b_{n-1}) \leq 1 \right) &= \frac{1}{n} \\
P(P_1 + b_1 \leq 1) \cdot \ldots \cdot P(P_{n-1} + b_{n-1} \leq 1) &= \frac{1}{n} \\
P(P_1 \leq 1 - b_1) \cdot \ldots \cdot P(P_{n-1} \leq 1 - b_{n-1}) &= \frac{1}{n} \\
\prod_{i=1}^{n-1} \frac{1 - b_i - l_i}{h_i - l_i} &= \frac{1}{n}
\end{align*}
$$

This is satisfied if $\forall i$:

$$
\begin{align*}
\frac{1 - b_i - l_i}{h_i - l_i} &= \left(\frac{1}{n}\right)^{\frac{1}{n-1}} \\
1 - b_i - l_i &= \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (h_i - l_i) \\
b_i &= 1 - l_i - \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (h_i - l_i)
\end{align*}
$$

Hence, $b_i = 1 - l_i - \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (h_i - l_i)$ is a solution.

In the vector notation of the biasing function we would therefore write:

$$
\mathbf{f}(\mathbf{l}, \mathbf{h}, r) = 1 - \mathbf{l} - \left(\frac{1}{n}\right)^\frac{1}{n-1}(\mathbf{h} - \mathbf{l})
$$

For the readers coming from [Carlini et al., 2024](#carlini2024) we note that the solution can be rewritten using an equivalent scaling term:

$$
\left(\frac{1}{n}\right)^\frac{1}{n-1} = \exp\left(\frac{-\log(n)}{n - 1}\right)
$$

### Finding parameters for normal distribution

In the previous derivation, we assumed a uniform distribution for all logits in the full output token vector. In this section, we hypothesize that logits follow a normal distribution.

We collected 52 logit distributions from the LLaMA-7b model for single character prompts, specifically using characters of the alphabet A-Z (26) and a-z (26). Each logit distribution was normalized with an assumed width of 40, and then the 52 logit distributions were flattened into a single vector. In Figure 1, we plot the histogram of the logit distribution with 100 bins in the interval $[0, 1]$.

![Logit Distribution Fitting](/assets/gaussianfit.png)

As seen in Figure 1, the logits can be reasonably well modeled using a normal distribution approximation. For our numerical experiments, we collected $\mu = 0.688$ and $\sigma = 0.066$, which we use extensively in the following tests. We note that while the following algorithms assume knowledge of the $\mu$ and $\sigma$ of the logit distributions we are trying to steal, in practice, these parameters can be inferred by either transferring them from smaller models or estimating them from a small number of logit distributions already collected.

### StartOverN With Normal Distribution Prior

Now we can formalize an algorithm using the normal distribution prior. We consider each logit as a random variable drawn from a normal distribution using the parameters $\mu$ and $\sigma^2$, which we have derived in previous section. At each step of the algorithm we bound the logit $z_i$ using $l_i$ and $h_i$ meaning the logits are actually drawn from a truncated normal distribution over the interval $[l_i, h_i]$.

Formally, we note

Let $P_i \sim N(\mu, \sigma^2)$ be truncated such that $l_i \leq P_i \leq h_i$ and $0 \leq l_i \leq h_i \leq 1$, where $\Phi$ is the CDF of $N(\mu, \sigma^2)$.

Then, the CDF of $P_i$ is given by:

$$
\Phi_{P_i}(x) = P( P_i \leq x ) = \frac{\Phi(x) - \Phi(l_i)}{\Phi(h_i) - \Phi(l_i)}
$$

**Theorem 2:** Given independent and identically distributed (i.i.d.) random variables $P_1, P_2, \ldots, P_{n-1}$ from a truncated normal distribution with truncation points $l_i$ and $h_i$, and constants $b_1, b_2, \ldots, b_{n-1}$, the StartOverN algorithm aims to find $b_i$ such that:

$$
P \left( \max(P_1 + b_1, P_2 + b_2, \ldots, P_{n-1} + b_{n-1}) \leq 1 \right) = \frac{1}{n}
$$

A solution to this problem is given by:

$$
b_i = 1 - \Phi^{-1} \left( \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (\Phi(h_i) - \Phi(l_i)) + \Phi(l_i) \right)
$$

**Proof:** We need

$$
\begin{align*}
P \left( \max(P_1 + b_1, \ldots, P_{n-1} + b_{n-1}) \leq 1 \right) &= \frac{1}{n} \\
P(P_1 + b_1 \leq 1) \cdot \ldots \cdot P(P_{n-1} + b_{n-1} \leq 1) &= \frac{1}{n} \\
P(P_1 \leq 1 - b_1) \cdot \ldots \cdot P(P_{n-1} \leq 1 - b_{n-1}) &= \frac{1}{n} \\
\prod_{i=1}^{n-1} \frac{\Phi(1 - b_i) - \Phi(l_i)}{\Phi(h_i) - \Phi(l_i)} &= \frac{1}{n} \\
\end{align*}
$$

This is satisfied if $\forall i$:

$$
\begin{align*}
\frac{\Phi(1 - b_i) - \Phi(l_i)}{\Phi(h_i) - \Phi(l_i)} &= \left(\frac{1}{n}\right)^{\frac{1}{n-1}} \\
\Phi(1 - b_i) - \Phi(l_i) &= \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (\Phi(h_i) - \Phi(l_i)) \\
\Phi(1 - b_i) &= \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (\Phi(h_i) - \Phi(l_i)) + \Phi(l_i) \\
1 - b_i &= \Phi^{-1} \left( \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (\Phi(h_i) - \Phi(l_i)) + \Phi(l_i) \right) \\
b_i &= 1 - \Phi^{-1} \left( \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (\Phi(h_i) - \Phi(l_i)) + \Phi(l_i) \right)
\end{align*}
$$

Hence, $b_i = 1 - \Phi^{-1} \left( \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (\Phi(h_i) - \Phi(l_i)) + \Phi(l_i) \right)$ is a solution.

We have derived the bias term in similar fashion to the uniform prior. Again we write the vector notation of the biasing function as:

$$
\mathbf{f}(\mathbf{l}, \mathbf{h}, r) =  1 - \Phi^{-1} \left( \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (\Phi(\mathbf{h}) - \Phi(\mathbf{l})) + \Phi(\mathbf{l}) \right).
$$

The generalized form of the function for any distribution truncated on the interval $[a, b]$, where the CDF is defined as:

$$
F_{[a,b]}(x) = \frac{F(x) - F(a)}{F(b) - F(a)}, \quad \text{for} \; a \leq x \leq b.
$$

This can be analogously formulated as:

$$
\mathbf{f}(\mathbf{l}, \mathbf{h}, r) = 1 - F^{-1} \left( \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (F(\mathbf{h}) - F(\mathbf{l})) + F(\mathbf{l}) \right)
$$

This result can be useful when considering different distributional assumptions for the logit

## Numerical analysis

Notice that at each iteration of Algorithm 1, we update the lower bound $\mathbf{l}$ and upper bound $\mathbf{h}$. Upon reaching a specified stopping condition, we return both bounds as the best estimates for all logits. This approach implies that, up to this point, we have not made any concrete _guess_ about the actual values of the logits. While a straightforward solution might involve taking the midpoint, i.e., $\frac{\mathbf{h} + \mathbf{l}}{2}$, as demonstrated in this section, this strategy may be suboptimal in scenarios where the attacker is constrained by a finite query budget $T$.

**Definition:** Let $\mathbf{l}, \mathbf{h} \in \mathbb{R}^{N_b}$ be vectors representing the lower and upper bounds, respectively. A **guess** is a function $g: \mathbb{R}^{N_b} \times \mathbb{R}^{N_b} \to \mathbb{R}^{N_b}$ that maps $\mathbf{l}$ and $\mathbf{h}$ to a vector $\mathbf{g} \in \mathbb{R}^{N_b}$ such that $\mathbf{l} \leq \mathbf{g} \leq \mathbf{h}$ element-wise.

We have already discussed _errors_ that were defined as the $l_1-norm$ and $l_{\infty}-norm$ of the bounds, that is $$\| \mathbf{h} - \mathbf{l} \|_1$$ and $$\| \mathbf{h} - \mathbf{l} \|_{\infty}$$ respectively. For testing purposes of how our algorithms are performing we can analogously define an error in terms of the true value of the logits $z$ and our $guess$, similarly we consider $$\| \mathbf{z} - \mathbf{g} \|_2$$.

In all results we also report on the performance of the simple _simultaneous-binary-search_ algorithm, which is defined by the biasing function $\mathbf{f}(\mathbf{l}, \mathbf{h}, r) = 1 - \frac{\mathbf{l} + \mathbf{h}}{2}$.

Using $$\| \mathbf{h} - \mathbf{l} \|_1$$ and $$\| \mathbf{z} - \mathbf{g}_m \|_2$$, we evaluate the performance of the _simultaneous-binary-search_, _start-over-n-with-uniform-prior_, and _start-over-n-with-normal-distribution_ algorithms on the 10 logit distributions we have collected from LLaMa-7b, as shown in [Figure 1](#figure-1). We assess the algorithms using the mid guess, defined as $\mathbf{g}_m = \frac{\mathbf{h} + \mathbf{l}}{2}$, and plot the logarithm of the error. To illustrate an interesting phenomenon, we draw error lines for 10 different logit vectors for each algorithm, instead of reporting confidence intervals.

<a id="figure-1"></a>
![Visualization of algorithms performance - L2 Norm](/assets/l1_bounds.png)

<div style="text-align: center; margin-bottom: 20px;"><em>Figure 1(a): Visualization of algorithms performance - $\| \mathbf{z} - \mathbf{g}_m \|_2$, $N_b = 50$</em></div>

![Visualization of algorithms performance - L1 Norm](/assets/l2normal.png)

<div style="text-align: center; margin-bottom: 20px;"><em>Figure 1(b): Visualization of algorithms performance - $\| \mathbf{h} - \mathbf{l} \|_1$, $N_b = 50$</em></div>

As observed in [Figure 1](#figure-1), our new algorithm, _start-over-n-with-normal-distribution_, outperforms both the state-of-the-art _start-over-n-with-uniform-prior_ and _simultaneous-binary-search_. Interestingly, the algorithm exhibits a 'jump' around the 100th iteration mark. To better understand this behavior, we inspect the bounds at the steps immediately before and after the jump, as shown in [Figure 2](#figure-2).

<a id="figure-2"></a>
![Example of logits at iteration 98 with blue g_m](/assets/it_98_mid_guess.png)

<div style="text-align: center"><em>Figure 2(a): Example of logits at iteration 98 with blue $\mathbf{g}_m$</em></div>
![Example of logits at iteration 99 with blue g_m](/assets/it_99_mid_guess.png)
<div style="text-align: center; margin-bottom: 20px;"><em>Figure 2(b): Example of logits at iteration 99 with blue $\mathbf{g}_m$</em></div>

As seen in the graphs in [Figure 2](#figure-2), only the upper bound of the last logit, $$z_{last}$$, was updated before the jump. Referencing Observation 1, if a logit $$z_i$$ has never been sampled, we can only update its upper bound $$h_i$$. This explains the observed jump. The algorithm spends the first 98 iterations before sampling the last token, $$z_{last}$$. Once the last token is sampled, it quickly establishes a precise lower bound, $$l_i$$. This behavior is analogous to the perfect algorithm described in Section 3.4, where at step $$N_b - 1$$, the lower bounds $$l_i$$ of all logits are close to the true values $$z_i$$, and at step $$N_b$$, all upper bounds $$h_i$$ are updated simultaneously.

This implies that for $$\| \mathbf{h} - \mathbf{l} \|_1$$, the error at iteration 98 is dominated by the difference $$h_{last} - l_{last}$$, and for $$\| \mathbf{z} - \mathbf{g}_m\|_2$$, by the difference $$z_{last} - g_{m_{last}}$$. Hence, we observe a dramatic change in the error graph.

One way to correct for jumps is to define a new _weird_ error metric that uses a combination of bounds and the actual values.

$$
weird(\mathbf{l}, \mathbf{h}, \mathbf{z}) = \sum_{i=1}^{N_b} \min(h_i - z_i, z_i - l_i)
$$

<a id="figure-3"></a>
![weird(l, h, z), N_b = 50](/assets/weird_best.png)

<div style="text-align: center; margin-bottom: 20px;"><em>Figure 3: $weird(\mathbf{l}, \mathbf{h}, \mathbf{z})$, $N_b = 50$</em></div>

As shown in [Figure 3](#figure-3), using the newly defined error metric, the jumps have disappeared. While this error metric addresses the problem, it has a drawback: if the attacker has a low budget of queries $$T$$, our new attack would underperform until about the 100th iteration.

A slightly better approach would be to adjust our guess by leveraging our understanding of the reason for the jump. One general solution is to weight how far the lower bound $$l_i$$ is from its initial value 0, and how far the upper bound $$h_i$$ is from its initial value 1. This ensures that we are closer to the respective bound in proportion to how far we are from the initial value. We achieve this with the following definition.

$$
\mathbf{m} = \frac{\mathbf{h} + \mathbf{l}}{2}
$$

$$
\mathbf{r} = \frac{\mathbf{h} - \mathbf{l}}{2}
$$

$$
\mathbf{g}_w = \mathbf{m} + \left(\frac{1 - \mathbf{h}}{\mathbf{l} + (1 - \mathbf{h})}\right) \mathbf{r} - \left(\frac{\mathbf{l}}{\mathbf{l} + (1 - \mathbf{h})}\right) \mathbf{r}
$$

We start by centering our guess at the middle value of the bounds, and then weighting how far we have moved from the initial value: $$1 - \mathbf{h}$$ for the upper bound and $$\mathbf{l}$$ for the lower bound, divided by the total amount we have moved from the initial values, $$\mathbf{l} + (1 - \mathbf{h})$$. Using these weights, we scale the radius $$\mathbf{r}$$ of our interval.

When we examine the state of our _start-over-n-with-normal-prior_ algorithm before sampling the last token, i.e., before the jump, we observe that the weighted guess is correctly aligned with its true value. This is because, for tokens that have not been sampled yet, the weighted error equals the upper bound $$h_i$$, which means $$\mathbf{g}_w = \mathbf{h}$$ for all unsampled tokens.

<a id="figure-4"></a>
![Example of logits at iteration 98 with blue g_m and red g_w](/assets/it_98_weighted_guess.png)

<div style="text-align: center; margin-bottom: 20px;"><em>Figure 4: Example of logits at iteration 98 with blue $\mathbf{g}_m$ and red $\mathbf{g}_w$</em></div>

Finally, we run the algorithms using the weighted guess $$\mathbf{g}_w$$ with $$\| \mathbf{z} - \mathbf{g}_w \|_2$$ to inspect its stability in [Figure 5](#figure-5). Using the weighted guess we are now on par with _start-over-n-with-uniform-prior_ during the first 100 iterations.

<a id="figure-5"></a>
![Visualization of algorithms performance - Weighted guess](/assets/l2weighted.png)

<div style="text-align: center; margin-bottom: 20px;"><em>Figure 5: Visualization of algorithms performance - $\| \mathbf{z} - \mathbf{g}_w \|_2$, $N_b = 50$</em></div>
We hypothesize that error jumps are an essential characteristic of highly performant algorithms, as they indicate precise boundary estimation.

## Performance

We evaluate the performance of various algorithms designed to extract logit distributions from large language models. We compare the algorithms based on the number of queries required to achieve a specified precision and report on the queries per logit metric. We measure their performance using the l2-norm of the difference between the true logit vector $$\mathbf{z}$$ and the weighted guess $$\mathbf{g}_w$$ by $$\| \mathbf{z} - \mathbf{g}_w \|_2$$.

| Name                               | Bias Function $$\mathbf{f}(\mathbf{l}, \mathbf{h})$$                                                                               | Queries per logit |
| ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- | ----------------- |
| Simultaneous Binary Search         | $$1 - \frac{\mathbf{l} + \mathbf{h}}{2}$$                                                                                          | 3.12              |
| Start Over N with Uniform Prior    | $$1 - \mathbf{l} - \left(\frac{1}{n}\right)^\frac{1}{n-1}(\mathbf{h} - \mathbf{l})$$                                               | 2.42              |
| **Start Over N with Normal Prior** | $$1 - \Phi^{-1} \left( \left(\frac{1}{n}\right)^{\frac{1}{n-1}} (\Phi(\mathbf{h}) - \Phi(\mathbf{l})) + \Phi(\mathbf{l}) \right)$$ | **1.98**          |

Table 1: Comparison of Algorithms with Precision $10^{-6}$ and $\mathbf{g}_w$

## References

<a id="carlini2024"></a>Carlini, N., Paleka, D., Dvijotham, K. D., Steinke, T., Hayase, J., Cooper, A. F., Lee, K., Jagielski, M., Nasr, M., Conmy, A., Wallace, E., Rolnick, D., & Tramèr, F. (2024, March). Stealing Part of a Production Language Model [arXiv:2403.06634 [cs]]. Retrieved May 1, 2024, from [http://arxiv.org/abs/2403.06634](http://arxiv.org/abs/2403.06634)
