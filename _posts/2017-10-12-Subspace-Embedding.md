---
layout: post
title: "Subspace-Embedding"
permalink: /subspace-embedding
date: 2017-10-10
categories: ['Theory', 'Machine Learning']
author: Junxiong Wang
authorlink: http://ovss.github.io
tags: [Theory]
---

Subspace embedding is a powerful tool to simplify the matrix calculation and analyze high dimensional data, especially for sparse matrix. 

## Subspace Embedding
A random matrix $\Pi \in \mathbb{R}^{m \times n}$ is a $(d, \epsilon, \delta)$-subspace embedding if for every $d$-dimensional subspace $U \subseteq \mathbb{R}^n$, $\forall x \in U$ has,

$$
\mathrm{P}(\left| ||\Pi x||_2 -  ||x||_2 \right| \leq \epsilon ||x||_2) \geq 1 - \delta
$$

Essentially, the sketch matrix maps any vector $x \in \mathbb{R}^n$ in the span of the columns of $U$ to $\mathbb{R}^m$ and the $l_2$ norm is preserved with high probability.

## Matrix Multiplication via Subspace Embedding
Consider a simple problem, given two matrix $A, B \in \mathbb{R}^{n \times d}$, what is the complexity to compute the  $C = A^{\top} B$? The simple algorithm takes $O(nd^2)$. Now we use subspace embedding to solve it. The result matrix is just $C' = (\Pi A)^{\top} (\Pi B)$. We can prove that with at least $1 - 3d^2 \epsilon$ probability, $\| C' - C \|_F \leq \epsilon \| A \|_F \| B \|_F $ holds. 

## Least Squares Regression via Subspace Embedding

Before we introduce subspace embedding, consider a simple problem, least squares regression. The exact least squares regression is the following problem: Given $A \in \mathbb{R}^{n \times d}$ and $b \in \mathbb{R}^n$, solve that

$$
x^{*} = \arg \min_{x \in R^d } \| Ax - b  \|_2 \qquad (1)
$$

It is well-known that the solution is $(A^{\top}A)^{+} A^T b$, where $(A^{\top}A)^{+}$ is the Moore-Penrose pseudoinverse of $A^{\top}A$. It can be calculated via SVD computation, taking $O(n d^2)$ time. However, if we allow approximation, can we decrease the time complexity? We can formalize the question as below, instead of finding the exact solution $x^{*}$, we would like to find $x' \in \mathbb{R}^d$ such that,

$$
\| Ax^{*} - b  \|_2 \leq  \| Ax' - b  \|_2  \leq (1 + \Delta) \| Ax^{*} - b  \|_2
$$

where $\Delta$ is a small constant number.

Suppose there exist a $(d+1, \epsilon, \delta)$-subspace embedding matrix $\Pi$, can we solve the following problem instead.

$$
x' = \arg \min_{x \in R^d } \| \Pi Ax - \Pi b  \|_2 \qquad (2)
$$

Proof: By the definition of $d+1$-subspace embedding matrix, the following equation holds with probability at least $1 - \delta$ for every arbitrary $x \in \mathbb{R}^d$

$$
\left| \| \Pi [A;b] \cdot [x^{\top}; -1] \|_2 - \| [A;b] \cdot [x^{\top}; -1] \|_2 \right| \leq \epsilon \| [A;b] \cdot [x^{\top}; -1] \|_2 \qquad (3)
$$


For $x'$ is optimum in equation(2), we have

$$
\begin{align}
\| \Pi Ax' - \Pi b \|_2 \leq \| \Pi Ax^{*} - \Pi b \|_2 
\end{align} \qquad (4)
$$

Replace $x^{\star}$ with $x$ in equation(3), we have

$$
\| \Pi Ax^{\star} - \Pi b  \|_2 \leq (1 + \epsilon) \|Ax^{\star} - b \|_2 \qquad (5)
$$

Replace $x'$ with $x$ in equation(3), we have

$$
\| \Pi Ax' - \Pi b  \|_2 \geq (1 - \epsilon) \|Ax' - b \|_2 \qquad (6)
$$

Combine equation(4, 5, 6) to get

$$
(1 - \epsilon) \|Ax' - b \|_2 \leq (1 + \epsilon) \|Ax^{\star} - b \|_2
$$

Take $\Delta = \frac{2 \epsilon}{1 - \epsilon}$ to conclude that the solution in equation(2) satisfies the desired statement.

$$
\| Ax^{*} - b  \|_2 \leq  \| Ax' - b  \|_2  \leq (1 + \Delta) \| Ax^{*} - b  \|_2
$$

Util now, we have seen how to solve approximate least regression problem by subspace embedding. However, one fundamental questions may arise, how to construct subspace embedding matrix? In the following section, we demonstrate that CountSketch is a subspace embedding. 


## Subspace Embedding Via CountSketch
CountSketch matrix $S \in \mathbb{R}^{B \times n}$ is defined as follows, fix the number of buckets $B$, a hash function $h:[n] \rightarrow [B]$ and a sign function $\phi:[n] \rightarrow \{-1, +1\}$. For $r \in [B], a \in [n]$, let

$$
S_{ra} = \begin{cases}
				\phi(a) & \text{if } h(a) = r \\
				0 & \text{otherwise}
			\end{cases}
$$

CountSketch Example:

$$
\left(
\begin{matrix}
0 & -1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
1 & 0 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & -1 & 0 & 0 & 0 & 1 & 0 & 1 \\
0 & 0 & 0 & 1 & 0 & 0 & -1 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & -1 & 0 & 0 & 0 \\
\end{matrix}
\right)
$$

We can show that for every subspace $U \in \mathbb{R}^{n \times d}$, then

$$
P(\left| ||\Pi x||_2 -  ||x||_2 \right| \leq \epsilon ||x||_2, \forall x \in \text{the column span of }U) > 1 - \delta
$$

Proof:

For $x$ is the the column span of $U$, then write $x$ as $Uy$ where $y \in \mathbb{R}^d$.

$$
(1 - \epsilon)||x||_2^2 \leq ||\Pi x||_2^2 \leq (1+\epsilon)||x||_2^2
$$

equivalent to 

$$
(1 - \epsilon) y^{\top} U^{\top} U y \leq y^{\top} U^{\top} \Pi^{\top} \Pi U y \leq (1+\epsilon) y^{\top} U^{\top} U y 
$$

For $U^{\top} U = I$,

$$
(1 - \epsilon) y^{\top} y \leq y^{\top} U^{\top} \Pi^{\top} \Pi U y \leq (1+\epsilon) y^{\top} y 
$$

equivalent to 

$$
|| U^{\top} \Pi^{\top} \Pi U - I ||_2 \leq \epsilon
$$

Since Frobenius norm upper bounds spectral norm, it suffices to show that

$$
|| U^{\top} \Pi^{\top} \Pi U - I ||_F \leq \epsilon
$$

We can show that (the detailed proof is ignored)

$$
\mathbf{E}[ ||U^{\top} \Pi^{\top} \Pi U - I ||_F^2 ] \leq \frac{2 d^2}{B}
$$

By the Markov’s inequality,

$$
\mathrm{P}( ||U^{\top} \Pi^{\top} \Pi U - I ||_F^2 
\geq \epsilon^2) \leq \frac{2 d^2}{B \epsilon^2}
$$ 

Then we can obtain
$$
\mathrm{P}( ||U^{\top} \Pi^{\top} \Pi U - I ||_F 
\geq \epsilon) \leq \frac{2 d^2}{B \epsilon^2}
$$

Thus 

$$
\mathrm{P} (\left| ||\Pi x||_2 -  ||x||_2 \right| \leq \epsilon ||x||_2) \geq 1 -  \frac{2 d^2}{B \epsilon^2}
$$

which implies that CountSketch is a $(d, \epsilon, \frac{2 d^2}{B \epsilon^2})$-subspace embedding. Setting $B = \frac{C d^2}{\epsilon^2}$ for a large enough absolute constant $C$ gives a subspace embedding with large constant probability.

## Complexity Analysis
The matrix $\Pi A$ is a $B \times d$ matrix, where $B = \frac{C d^2}{\epsilon^2}$. Thus using SVD to solve $||\Pi A x - \Pi b||$ takes $ploy(d, \frac{1}{\epsilon})$. How much time does it take to form the matrix $\Pi A$ and the vector $\Pi b$? Since every column of $\Pi$ has exactly one nonzero, the runtime of this is proportional to the number of nonzeros in the matrix $A$ and the vector $b$. The overall time is $O(nnz(A) + ploy(d, \frac{1}{\epsilon}))$. Note that if the matrix is sparse, this is very efficient.

## Experiment


# Reference
- EPFL Topics in Theoretical Computer Science (Sublinear Algorithm for Big Data Analysis), 2017

- Xiangrui Meng and Michael W. Mahoney. Low-distortion subspace embeddings in input-sparsity
time and applications to robust linear regression, 2012.

- Jelani Nelson and Huy L. Nguyen. Osnap: Faster numerical linear algebra algorithms via sparser
subspace embeddings, 2012.

