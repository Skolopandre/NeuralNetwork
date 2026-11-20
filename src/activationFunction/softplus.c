#include "activation.h"
#include <math.h>

float softplus_activation(float x) {
    return log(1.0 + exp(x)); ;
}
float softplus_derivative(float x) {
    if (x >= 0.0) {
        return 1.0 / (1.0 + exp(-x));
    } else {
        double exp_x = exp(x);
        return exp_x / (1.0 + exp_x);
    }
}
const ActivationFunction SOFTPLUS = {
    .activation = softplus_activation,
    .derivative = softplus_derivative
};