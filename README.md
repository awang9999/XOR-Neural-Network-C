# XOR-Neural-Network-C
An implementation of a simple neural network that replicates a fuzzy XOR behavior.

Project: C Neural Network that learns XOR
Author: Alexander Wang (aw576@cornell.edu/alexander.wang2001@gmail.com)
Last updated: 2020-08-06

Description:
In this project, I will attempt to recreate the popular
simple neural network that learns fuzzy XOR behavior from scratch in
the C language. I use a matrix data structure to represent the nodes, weights, 
and biases. The forward propagation and back propagation code will be guided by 
Santiago Becerra's article about neural networks in C.
(https://towardsdatascience.com/simple-neural-network-implementation-in-c-663f51447547).

This neural network will consist of 1 hidden layer with two hidden nodes. This
will be the simplest architecture capable of learning the fuzzy XOR behavior.

Expected XOR Behavior:
A    B    Output
----------------
0    0       0
0    1       1
1    0       1
1    1       0

Planned Activation functions for each layer:
Input layer: identity
Hidden layer: tanh
Output layer: sigmoid
