#include "activation.h"

float linear_activation(float x) {
    return x ;
}
float linear_derivative(float x) {
    return x ;
}
const ActivationFunction LINEAR = {
    .activation = linear_activation,
    .derivative = linear_derivative
};
