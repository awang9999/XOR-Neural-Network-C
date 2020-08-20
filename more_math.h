#ifndef MORE_MATH_H
#define MORE_MATH_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

double abs_val(double x);

bool double_equals(double a, double b);

double sigmoid(double x);

double d_sigmoid(double x);

double d_tanh(double x);

double rand_weight();

double logistic_1(double x);

double identity_1(double x);

#endif