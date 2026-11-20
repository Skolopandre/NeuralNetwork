#ifndef NETWORK_H
#define NETWORK_H

#include "optimizer.h"
#include "layer.h"
#include "network_builder.h"

typedef struct network_builder network_builder;
typedef network_builder* network_builder_t;

typedef struct Network Network;
typedef Network* Network_t;

//-----------------initialisation-----------------
Network_t init_network(network_builder_t builder, OptimizerType optim_type);
void free_network(Network_t NN);


//-----------------manipulation------------------

void network_reset_reservoir(Network_t NN);

// PRE : input.size = NN.layer1.size
void forward_propagation(Network_t NN, const float *input);
void backpropagation(Network_t NN,const float* restrict input, const float *restrict target);
size_t predict(Network_t NN);
void predict_full(Network_t NN, const float *input, float *output);
float mse_loss(Network_t NN, const float *target);

//-----------------Sauvegarde des poids-----------
void save_weights(Network_t NN, const char *filename);
void load_weights(Network_t NN, const char *filename);


//-----------------Getter/Setter-----------------
size_t network_get_num_layers(Network_t NN);
Layer_t network_get_layer(Network_t NN, size_t index);
Optimizer_t network_get_optimizer(Network_t NN);
size_t network_get_input_size(Network_t NN);
size_t network_get_output_size(Network_t NN);
void network_set_num_layers(Network_t NN, size_t num_layers);
void network_set_num_layers(Network_t NN, size_t num_layers);
void network_set_layers(Network_t NN,Layer_t* layers);
void network_set_input_size(Network_t NN, size_t input_size);
void network_set_output_size(Network_t NN, size_t output_size);
void network_set_opt(Network_t NN, Optimizer_t optimizer);
void network_set_grad_a(Network_t NN, void* ptr);
void network_set_grad_b(Network_t NN, void* ptr);
void network_set_gradient_capacity(Network_t NN, size_t capacity);

#endif // NETWORK_H