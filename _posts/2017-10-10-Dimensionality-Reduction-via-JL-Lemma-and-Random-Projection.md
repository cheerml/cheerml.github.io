---
layout: post
title: "Dimensionality Reduction via JL Lemma and Random Projection"
permalink: /random-projection
date: 2017-10-10
categories: ['Theory', 'Machine Learning']
author: Junxiong Wang
authorlink: http://ovss.github.io
tags: [Theory]
---


Nowadays, dimensionality is a serious problem of data analysis as the huge data we experience today results in very sparse sets and very high dimensions. Although, data scientists have long used tools such as principal component analysis (PCA) and independent component analysis (ICA) to project the high-dimensional data onto a subspace, but all those techniques reply on the computation of the eigenvectors of a $n \times n$ matrix, a very expensive operation (e.g., spectral decomposition) for high dimension $n$. Moreover, even though eigenspace has many important properties, it does not lead good approximations for many useful measures such as vector norms. We discuss another method random projection to reduce dimensionality.

In 1984, two mathematicians introduced and proved the following lemma.

## Johnson-Lindenstrauss lemma
For any $\epsilon \in (0,\frac{1}{2})$, $\forall x_1, x_2, \dots, x_d \in \mathrm{R}^{n}$, there exists a matrix $M \in \mathrm{R}^{m \times n}$ with $m = O(\frac{1}{\epsilon^2} \log{d})$ such that $\forall 1 \leq i,j \leq d$, we have

$$
(1-\epsilon)||x_i - x_j||_2 \leq ||Mx_i - Mx_j||_2 \leq (1+\epsilon)||x_i - x_j||_2 
$$
 
Remark: This lemma states that for any pair vector $x_i, x_j$ in $d$ dimension, there exist a sketch matrix $M$ which maps $\mathrm{R}^n \rightarrow \mathrm{R}^m$ and the Euclidean distance is preserved within $\epsilon$ factor. The result dimension does not have any relationship to origin dimension $n$ (only relates to the number of vector pairs $d$).

During a long time, no one can figure out how to get this sketch matrix.

## Random Projection
Until 2003, some researches point out that this sketch matrix can be created using Gaussian distribution.

Consider the following matrix $A \in \mathrm{R}^{m \times n}$, where $A_{ij} \sim \mathcal{N}(0,1)$ and all $A_{ij}$ are independent. We claim that this matrix satisfies the statement of JL lemma.

Proof. It is obvious that sketch has an additional property, 
$\forall i, (Ax)\_i = \sum\_{j=1}^{n} A\_{ij} x\_j \sim \mathcal{N}(0, ||x||_2^2)$. In other word, Gaussian distribution is 2-stable distribution. Then we can obtain $\|\|Ax\|\|\_2^2 = \sum\_{i=1}^{m} y\_i^2$, where $y\_i \sim \mathcal{N}(0, \|\|x\|\|\_2^2)$. That is to say, $\|\|Ax\|\|\_2^2$ follows a $\chi^2$ (chi-squared) distribution with degrees of freedom $m$. For tail bound of $\chi^2$ distribution, we can get

$$
P(||Ax||_2^2 - m||X||_2^2| > \epsilon m||X||_2^2) < \exp(-C \epsilon^2 m)
$$
for a constant $C > 0$.

Fix two index $i, j$, and let $y^{ij} = x_i - x_j$ and $M = \frac{1}{\sqrt{m}} A$, and set $m = \frac{4}{C \epsilon^2} \log{n}$ to get

$$
P(|||M y^{ij}||_2^2 - ||y^{ij}||_2^2| > \epsilon ||y^{ij}||_2^2) < \exp(-C \epsilon^2 m) = \frac{1}{n^4}
$$

Take the union bound to obtain, 

$$
P(\forall i \neq j, |||M y^{ij}||_2^2 - ||y^{ij}||_2^2| > \epsilon ||y^{ij}||_2^2 ) < \sum_{i \neq j} P(|||M y^{ij}||_2^2 - ||y^{ij}||_2^2| < \epsilon ||y^{ij}||_2^2 ) < {n \choose 2} n^4 < \frac{1}{n^2}
$$

which is same as the guarantee in Johnson-Lindenstrauss lemma.

## Application
In this or other forms, the JL lemma has been used for a large variety of computational tasks, especially in streaming algorithm, such as

* [Computing a low-rank approximation to the original matrix A.](https://www.stat.berkeley.edu/~mmahoney/f13-stat260-cs294/Lectures/lecture19.pdf)

* [Finding nearest neighbors in high-dimensional space.](http://web.stanford.edu/class/cs369g/files/lectures/lec16.pdf)

* [Simplify the calculation of the effective resistance in Graph Spectral Sparsification.](https://simons.berkeley.edu/sites/default/files/docs/1768/slidessrivastava1.pdf)

* [Relates to Graph Sketches.](https://people.cs.umass.edu/~mcgregor/papers/12-pods1.pdf)

## Reference
- EPFL sublinear algorithm course, 2017
- EPFL advanced algorithm course, 2016
- Johnson, William B.; Lindenstrauss, Joram (1984). "Extensions of Lipschitz mappings into a Hilbert space". In Beals, Richard; Beck, Anatole; Bellow, Alexandra; et al. Conference in modern analysis and probability (New Haven, Conn., 1982). Contemporary Mathematics. 26. Providence, RI: American Mathematical Society. pp. 189–206.
- Kane, Daniel M.; Nelson, Jelani (2012). "Sparser Johnson-Lindenstrauss Transforms". Proceedings of the Twenty-Third Annual ACM-SIAM Symposium on Discrete Algorithms,. New York: Association for Computing Machinery (ACM).


The post is used for study purpose only.

