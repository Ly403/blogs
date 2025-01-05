# Elucidating the Design Space of Diffusion-Based Generative Models

![image-20240918153327788](Elucidating the Design Space of Diffusion-Based Generative Models.assets/image-20240918153327788.png)



arxiv: https://doi.org/10.48550/arXiv.2206.00364

## 前言

这篇文章我看了很久了，也阅读了很多遍。说实话我觉得这篇文章理解起来是很有难度的，一来文章很长，加附录总共47页；二来附录里面有非常多的数学推导，倘若跳过不看就难以全面理解正文的内容。但是这篇文章（简称EDM）对后续扩散模型（尤其是连续型扩散模型）的发展起了非常大的推动作用。如果想要了解扩散模型原理，这篇文章几乎不可能不看。

因为我的工作和这篇文章比较相关，因此想通过写点笔记总结一下这篇文章，顺序上会按照我的理解重新组织，一方面加强理解和记忆，一方面可以作为博客发出去，故有此文。此外，我准备结合代码总结这篇文章，代码来自于https://github.com/Stability-AI/generative-models，来自stability-ai，代码的质量比较高，适合学习。我每天写一点，争取早日写完。

由于我也是刚学习扩散模型，水平所限，所以可能有些理解未必准确，如有错误还望指出。（套个盾）

## 一个统一的框架

这一节来自原文的附录B，这段我觉得非常重要，原文把推导放在附录里面了，但是最好还是要了解。

Song Yang博士在他2020年那篇著名的score-based generative model[^1]里面统一了连续型的扩散模型（NCSN[^2]，也称VE类型的扩散模型）和离散型的扩散模型（DDPM[^3]，也称VP类型的扩散模型），用一个统一的**随机微分方程（stochastic differential equations，SDE）**描述了扩散模型对图像进行前向加噪的过程：
$$
\mathrm{d}x =f(x,t)\mathrm{d}t+g(t)\mathrm{d}\omega_t \tag{1}
$$
$x$是代表图像随机过程，准确的表示应该是$x_t$。$t=0$时，$x_t$无噪声，随着$t$逐渐增大，噪声逐渐加入到$x_t$中。$\omega_t$是标准布朗运动（也称维纳过程），$f(\cdot,t) : R^d \rightarrow R^d$和$g(\cdot) : R \rightarrow R$分别是漂移系数和扩散系数。

但是（1）式还是太宽泛了，说白了这个框架里面$f(x,t)$和$g(t)$能有很多种选择，但我们实际使用的时候一般选择的$f(x,t)$和$g(t)$的形式不会特别复杂。**具体而言，我们一般选择$f(x,t)=f(t)x$。**基于此，可以对（1）简化：
$$
\mathrm{d}x =f(t)x\mathrm{d}t+g(t)\mathrm{d}\omega_t \tag{2}
$$

然后，重点来了。我们非常想知道在任意$t$时刻下$x_t$的具体情况，最好就是知道$x_t$的分布。因为$t$确定后，$x_t$由随机过程退化成随机变量，了解一个随机变量的分布就能全面把握它的信息。





[^1]: SONG Y, SOHL-DICKSTEIN J, KINGMA DiederikP, et al. Score-Based Generative Modeling through Stochastic Differential Equations[J]. Cornell University - arXiv,Cornell University - arXiv, 2020.
[^2]: Song Y, Ermon S. Generative modeling by estimating gradients of the data distribution[J]. Advances in neural information processing systems, 2019, 32.
[^3]: Ho J, Jain A, Abbeel P. Denoising diffusion probabilistic models[J]. Advances in neural information processing systems, 2020, 33: 6840-6851.

