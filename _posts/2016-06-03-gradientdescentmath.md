---
layout: post
title: "The math behind Gradient Descent"
permalink: /mathGD
date: 2016-09-06
categories: ['Machine Learning']
author: Jun Lu
authorlink: http://www.junlulocky.com
tags: [Learning theory]
---

This post means to help starters to understand the math behind Gradient Descent (GD).

## Intuitive understanding
An intuitive way to think of Gradient Descent is to imagine the path of a river originating from top of a mountain. The goal of gradient descent is exactly what the river strives to achieve - namely, reach the bottom most point (at the foothill) climbing down from the mountain. That's what you taught by your machine learning teacher right? But do you only understand this and use gradient descent naively every time? This post will help you understand the math behind gradient descent.

## Math behind Gradient Descent
Here I define the objective function to be \\(L(x,w,b)\\) and the input variable of \\(L\\) is \\(x\\) with \\(d\\)-dimension, weight variable \\(w\\) and bias variable\\(b\\), our goal is to use algorithm to get the minimum of \\(L(x,w,b)\\).

To make this question more precise, let's think about what happens when we move the ball a small amount \\(\Delta x_1\\) in the \\(x_1\\) direction, a small amount \\(\Delta x_2\\) in the \\(x_2\\) direction, ..., and a small amount \\(\Delta x_d\\) in the \\(x_d\\) direction. Calculus tells us that \\(L(x,w,b)\\) changes as follows:

$$\Delta L \approx \frac{\partial L}{\partial x_1}\Delta x_1 + ... + \frac{\partial L}{\partial x_d}\Delta x_d$$

In this sense, we need to find a way of choosing \\(\Delta x_1\\), ..., \\(\Delta x_d\\) so as to make \\(\Delta L\\) negative; i.e., we'll make the objective function decrease so that to minimize. 

- Define \\(\Delta x=(\Delta x_1, ..., \Delta x_d)^T\\) to be the vector of changes in \\(x\\). 
- Define \\(\nabla L=(\frac{\partial L}{\partial x_1}, ..., \frac{\partial L}{\partial x_d})^T\\) to be the gradient vector of \\(L\\). 

So we can find: $$\Delta L \approx \frac{\partial L}{\partial x_1}\Delta x_1 + ... + \frac{\partial L}{\partial x_d}\Delta x_d = \nabla L ^T \Delta x$$. By now, things are becoming easier. Suppose, \\(\Delta x=-\eta \nabla L\\) (i.e. the step size in gradient descent, where \\(\eta\\) is the learning rate). Then: 

$$\nabla L \approx -\eta \nabla L^T\nabla L = -\eta||\nabla L||_2^2 \leq 0$$

Now, we can find the rightness of gradient descent. We can use the following update rule to update next \\(x\\):

$$x^{k+1} = x^{k} - \eta \nabla L(x^k)$$

this update rule will make the objective function drop to the minimum point.

## Gradient Descent in a convex problem
Now, I will consider the gradient descent in a convex problem, because we usually use gradient descent in a convex problem, otherwise, we usually get the local minimum. If the objective function is convex, then \\(\nabla L(x^k)^T(x^{k+1}-x^{k})\geq 0\\) implies \\(L(x^{k+1}) \geq L(x^k)\\). This can be derived from the convex property of a convex function, i.e. \\(L(x^{k+1}) \geq L(x^k)^T(x^{k+1}-x^k)\\). 

In this sense, we need to make \\(\nabla L(x^k)^T(x^{k+1}-x^{k})\leq 0\\) so as to make the objective function decrease. In gradient descent \\(\Delta x\\) is chosen to be \\(-\nabla L(x^k)\\). However, there are many other descent method, such as **steepest descend**, **normalized steepest descent**, **newton step** and so on. The main idea of these methods is to make \\(\nabla L(x^k)^T(x^{k+1}-x^{k})= \nabla L^T \Delta x \leq 0\\).

## References
- Michael A. Nielsen, *Neural Networks and Deep Learning*, Determination Press, 2015
- Stephen Boyd, and Lieven Vandenberghe. *Convex optimization*. Cambridge university press, 2004.

## Remarks
Last updated on June 14, 2016

{% if site.url contains site.testurl1 %}
<div style="display:none;">
<center>
<a href="http://www.clustrmaps.com/map/Junlulocky.com/machine-learning/2016/06/03/gradientdescentmath/" title="Visit tracker for Junlulocky.com/machine-learning/2016/06/03/gradientdescentmath/"><img src="//www.clustrmaps.com/map_v2.png?u=0YFz&d=dPqQX9JbVugBKkw8uA4Q9ME40uE-X7RKY5L1f9ATkig" /></a>
</center>
</div>
{% endif %}







