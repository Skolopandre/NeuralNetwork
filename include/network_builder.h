#ifndef NETWORK_BUILDER_H
#define NETWORK_BUILDER_H
#include <network.h>

#include "layer.h"
typedef struct Network Network;
typedef Network* Network_t;

typedef struct network_builder{
    size_t size;
    size_t capacity;
    Layer_t *layers;
}network_builder;
typedef network_builder* network_builder_t;

network_builder_t builder_init(size_t input_size);

void builder_destroy(network_builder_t builder);

void builder_add_dense(network_builder_t builder,size_t size,ActivationFunction act);
void builder_add_reservoir(network_builder_t builder,
    size_t size,
    ActivationFunction act,
    float leak_rate,
    float average_density,
    float spectral_radius);


#endif //NETWORK_BUILDER_H