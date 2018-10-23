# Bayesian Neural Networks for Regression


A Tensorflow implementation of safe f-divergence for Bayesian neural network. 


### Using the code

Example:

`python bnn_trainer.py --method adapted --alpha -1.0 --n_epoches 500 --dataset boston --batch_size 32 --sample_size 100 --n_hidden 50 --trial 1 --learning_rate 1e-3 --save`

Use `run.sh` to reproduce our results in Table 1.


### References
Our code is based on the implementation of [Variational Renyi Bound](https://github.com/YingzhenLi/VRbound).

