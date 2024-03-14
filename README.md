# 1dnn

Code for our paper "A Library of Mirrors: Deep Neural Nets in Low Dimensions are Convex Lasso Models with Reflection Features" (https://arxiv.org/abs/2403.01046#). This paper shows that training neural networks with 1-D data is equivalent to solving a Lasso problem with an explicit and simple dictionary.

The training problem for a ReLU neural network $f(\theta;{\mathbf{X}})$ parameterized by $\theta$, trained on a 1-D training matrix $\mathbf{X} \in \mathbb{R}^N$ consisting of $N$ samples, and label vector $\mathbf{y}\in \mathbb{R}^N$, is

$ \min_{\theta \in \Theta}  \frac{1}{2} \|| f(\theta;{\mathbf{X}})  - \mathbf{y} \||^2_2 + \frac{\beta}{L}  \||{\theta}_w||_{L}^{L} $.

The weights are regularized with the $l_L$ penalty $ \||{\theta}_w||_{L}^{L} $.  

We show this non-convex training problem is equivalent to the Lasso problem

$$ \min_{\mathbf{z}, \xi}  \frac{1}{2} \| \mathbf{A}{} \mathbf{z} + \xi \mathbf{1} - \mathbf{y} \|^2_2 + \beta \|\mathbf{z}\|_1$$.


The files ```Fig2.ipynb```, ```Fig3.ipynb```, ```Fig18.ipynb```, and ```Fig19.ipynb``` in the directory ```/code``` contain the code for Figures 2,3,18, and 19 in the paper. The files ```Nonconvex_training.ipynb``` and ```convex_training.ipynb``` contain code for training 1D ReLU networks using the conventional, non-convex training problem and our equivalent convex, Lasso problem, respectively.
