#include "activation.h"
#include <math.h>

float tanh_activation(float x) {
    return tanhf(x);
}
float tanh_derivative(float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}
const ActivationFunction TANH = {
    .activation = tanh_activation,
    .derivative = tanh_derivative
};
