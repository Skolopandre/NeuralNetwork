#include "activation.h"

float lrelu_activation(float x) {
    return x > 0 ? x : 0.01f * x;
}
float lrelu_derivative(float x) {
    return x > 0 ? 1.0f : 0.01f;
}
const ActivationFunction LRELU = {
    .activation = lrelu_activation,
    .derivative = lrelu_derivative
};
