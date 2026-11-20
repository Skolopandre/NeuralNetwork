#include "activation.h"

float relu_activation(float x) {
    return x > 0 ? x : 0.0f;
}
float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}
const ActivationFunction RELU = {
    .activation = relu_activation,
    .derivative = relu_derivative
};
