## Confidence is All You Need for MI Attacks

This directory contains code to reproduce our paper:
**"Confidence is all you need for MI Attacks"** <br>
https://arxiv.org/abs/2311.15373 <br>
by Abhishek Sinha, Himanshi Tibrewal, Mansi Gupta, Nikhar Waghela, Shivank Garg

Our work is based upon : 

**"Membership Inference Attacks From First Principles"** <br>
https://arxiv.org/abs/2112.03570 <br>
by Nicholas Carlini, Steve Chien, Milad Nasr, Shuang Song, Andreas Terzis, and Florian Tramèr.

### INSTALLING DEPENDENCIES
To install the basic dependencies needed to run this repository 

>bash requirements.sh

We train our models with JAX + ObJAX so you will need to follow build instructions for that
https://github.com/google/objax
https://objax.readthedocs.io/en/latest/installation_setup.html

### RUNNING THE CODE

#### 1. Train the models

The first step in our attack is to train shadow models. As a baseline that
should give most of the gains in our attack, you should start by training 16
shadow models with the command

> bash scripts/train_demo.sh

or if you have multiple GPUs on your machine and want to train these models in
parallel, then modify and run

> bash scripts/train_demo_multigpu.sh

This will train several CIFAR-10 wide ResNet models to ~91% accuracy each, and
will output a bunch of files under the directory exp/cifar10 with structure:

```
exp/cifar10/
- experiment_N_of_16
-- hparams.json
-- keep.npy
-- ckpt/
--- 0000000100.npz
-- tb/
```

#### 2. Perform inference

Once the models are trained, now it's necessary to perform inference and save
the output features for each training example for each model in the dataset.

> python3 inference.py --logdir=exp/cifar10/

This will add to the experiment directory a new set of files

```
exp/cifar10/
- experiment_N_of_16
-- logits/
--- 0000000100.npy
```

where this new file has shape (50000, 10) and stores the model's output features
for each example.

#### 3. Compute membership inference scores

Finally we take the output features and generate our logit-scaled membership
inference scores for each example for each model.

> python3 score.py exp/cifar10/

We find the evaluation of scores through various experiments. The calculations of logits are implemented in the score.py file, where we explored all the commented-out calculations to find the logits. It was noted that utilizing argmax values, which doesn't require knowledge of true labels, produced results comparable to those outlined in the "LIRA Likelihood Ratio Paper."

And this in turn generates a new directory

```
exp/cifar10/
- experiment_N_of_16
-- scores/
--- 0000000100.npy
```

with shape (50000,) storing just our scores.

### PLOTTING THE RESULTS

Finally we can generate pretty pictures, and run the plotting code

> python3 plot.py

### RESULTS {Using AUC as Metric}

|         | Loss Value (Baseline) | Confidence Values | log (Confidence Values) | Argmax | log (Argmax) |
| :-----: | :-------------------: | :---------------: | :---------------------: | :----: | :----------: |
| Attack Ours (Online) | 0.5753 | 0.5668 | 0.575 | 0.5464 | 0.5447 |
| Attack Ours (Online,Fixed Variance) | 0.5879 | 0.593 | 0.6009 | 0.5622 | 0.5602 |
| Attack Ours (Offline) | 0.5181 | 0.492 | 0.4721 | 0.478 | 0.4756 | 
| Attack Ours (Offline, Fixed Variance) | 0.5184 | 0.4928 | 0.4804 | 0.4834 | 0.4815 |
| Attack Global Threshold | 0.5448 | 0.5439 | 0.5469 | 0..5376 | 0.5377 |

where the global threshold attack is the baseline, and our online,
online-with-fixed-variance, offline, and offline-with-fixed-variance attack
variants are the four other curves. Note that because we only train a few
models, the fixed variance variants perform best.

### Citation

You can cite this paper with

```
@ title= {Confidence is All You Need For MI Attacks}
  author={Abhishek Sinha, Himanshi Tibrewal, Mansi Gupta, Nikhar Waghela, Shivank Garg},
  journal={arXiv preprint arXiv:2311.15373},
  year={2023}
}
```