#ifndef ACTIVATION_H
#define ACTIVATION_H

typedef struct {
    float (*activation)(float);
    float (*derivative)(float);
} ActivationFunction;

/* Prototypes des fonctions */
float relu_activation(float x);
float relu_derivative(float x);

float tanh_activation(float x);
float tanh_derivative(float x);

float lrelu_activation(float x);
float lrelu_derivative(float x);

float linear_activation(float x);
float linear_derivative(float x);

float softplus_activation(float x);
float softplus_derivative(float x);

extern const ActivationFunction RELU;
extern const ActivationFunction TANH;
extern const ActivationFunction LRELU;
extern const ActivationFunction LINEAR;
extern const ActivationFunction SOFTPLUS;

#endif