# 1dnn

Code for our paper "A Library of Mirrors: Deep Neural Nets in Low Dimensions are Convex Lasso Models with Reflection Features" (https://arxiv.org/abs/2403.01046#). This paper shows that training neural networks with 1D data is equivalent to solving a Lasso problem with an explicit and simple dictionary.

In other words, for ReLu networks,

$	\min_{\params \in \Theta}  \frac{1}{2} \| \nn{L}{\mathbf{X}}  - \mathbf{y} \|^2_2 + \frac{\beta}{\effL{}}  \|\regparams\|_{\effL{}}^{\effL{}}$

The files ```Fig2.ipynb```, ```Fig3.ipynb```, ```Fig18.ipynb```, and ```Fig19.ipynb``` in the directory ```/code``` contain the code for Figures 2,3,18, and 19 in the paper. The files ```Nonconvex_training.ipynb``` and ```convex_training.ipynb``` contain code for training 1D ReLU networks using the conventional, non-convex training problem and our equivalent convex, Lasso problem, respectively.
