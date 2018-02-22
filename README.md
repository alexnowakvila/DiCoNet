# Divide and Conquer Networks (DiCoNet)

![figgeneral](imgs/figgeneral.png)
Code accompanying [Divide and Conquer Networks](https://arxiv.org/pdf/1611.02401.pdf)
# DiCoNet Summary
## Model

![eq1](imgs/eq1.png)

## Weak supervision Loss

![eq1](imgs/eq2.png)

## Gradients computation

![eq1](imgs/eq3.png)

# Reproduce Experiments
## Prerequisites

- Python 3.6.1 + some traditional libraries
- Computer with Linux or OSX
- [PyTorch](http://pytorch.org)
- For training, an NVIDIA GPU is needed. CPU not supported.

## Convex Hull
### Baseline
```
python code/ConvexHull2d/main.py --path [experiment folder] --path_dataset [dataset folder]
```
### Without split computational regularization
```
python code/ConvexHull2d/main.py --path [experiment folder] --path_dataset [dataset folder] --dynamic
```
### Add split computational regularization
```
python code/ConvexHull2d/main.py --path [experiment folder] --path_dataset [dataset folder] --dynamic --regularize_split
```

## K-means

Check the parse arguments section at the top of the kmeans.py file to change the default arguments.
```
python code/K-means/main.py --path [experiment folder] --path_dataset [dataset folder] --dataset ["GM"/"CIFAR"]
```

## Knapsack

Create a folder for the sinthetic dataset.
```
mkdir src/Knapsack/dataset
```
Compile the knapsack solver for the creation of the dataset.
```
g++ src/Knapsack/solver.cc -O2 -o src/Knapsack/solver
```
Train the model. Check the parse arguments section at the top of the knapsack.py file to change the default arguments.
```
python src/Knapsack/knapsack.py --dataset_path src/Knapsack/dataset --solver_path src/Knapsack/solver --logs_path src/Knapsack/
```


