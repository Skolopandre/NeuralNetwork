#include "network_builder.h"
#include "layer.h"
#include "aligned_alloc.h"

#define INITIAL_BUILDER_CAPACITY 4


network_builder_t builder_init(size_t input_size) {
    network_builder_t builder = malloc(sizeof(struct network_builder));
    if(!builder) {
        fprintf(stderr, "Failed to allocate memory for network_builder_t\n");
        return NULL;
    }

    builder->size = 1;
    builder->capacity = INITIAL_BUILDER_CAPACITY;
    builder->layers = malloc(builder->capacity * sizeof(struct Layer));
    if(!builder->layers) {
        fprintf(stderr, "Failed to allocate memory for layers\n");
        free(builder);
        return NULL;
    }

    Layer_t layer = create_layer_from_type(FLATTEN,input_size,0,TANH,0,0,0,NULL);      //tanh comme placeholder
    if(!layer) {
        fprintf(stderr, "Failed to create layer from type\n");
        free(builder);
        return NULL;
    }
    builder->layers[0] = layer;

    return builder;
}

void builder_destroy(network_builder_t builder) {
    if(builder) {
        if(builder->layers) {
            for(size_t i = 0; i < builder->size; i++) {
                layer_destroy(builder->layers[i]);
            }
        free(builder->layers);
        }
        free(builder);
    }
}

void builder_add_dense(network_builder_t builder,size_t size,ActivationFunction act) {
    if(!builder) {
        fprintf(stderr,"builder non-existant\n");
        return;
    }

    if(builder->size>=builder->capacity) {
        size_t new_capacity = builder->capacity+1;
        Layer_t* tmp = realloc(builder->layers,new_capacity*sizeof(struct Layer));
        if(!tmp) {
            fprintf(stderr, "Failed to allocate memory for layers\n");
            return;
        }
        builder->layers = tmp;
        builder->capacity = new_capacity;
    }

    Layer_t prev = builder->layers[builder->size-1];
    size_t input_size = prev->size;
    Layer_t layer = create_layer_from_type(DENSE,size,input_size,act,0,0,0,NULL);
    if(!layer) {
        fprintf(stderr, "Failed to create layer from type\n");
        return;
    }

    builder->layers[builder->size++] = layer;
}

void builder_add_reservoir(network_builder_t builder,
    size_t size,
    ActivationFunction act,
    float leak_rate,
    float average_density,
    float spectral_radius) {
    if(!builder) {
        fprintf(stderr,"builder non-existant\n");
        return;
    }
    if(builder->size>=builder->capacity) {
        size_t new_capacity = builder->capacity+1;
        Layer_t* tmp = realloc(builder->layers,new_capacity*sizeof(struct Layer));
        if(!tmp) {
            fprintf(stderr, "Failed to allocate memory for layers\n");
            return;
        }
        builder->layers = tmp;
        builder->capacity = new_capacity;
    }

    Layer_t prev = builder->layers[builder->size-1];
    size_t input_size = prev->size;
    Layer_t layer = create_layer_from_type(RESERVOIR,size,input_size,act,leak_rate,average_density,spectral_radius,NULL);
    if(!layer) {
        fprintf(stderr, "Failed to create layer from type\n");
        return;
    }

    builder->layers[builder->size++] = layer;
}


void builder_add_conv(network_builder_t builder,size_t size,ActivationFunction act,size_t* kernel_size) {
    if(!builder) {
        fprintf(stderr,"builder non-existant\n");
        return;
    }

    if(builder->size>=builder->capacity) {
        size_t new_capacity = builder->capacity+1;
        Layer_t* tmp = realloc(builder->layers,new_capacity*sizeof(struct Layer));
        if(!tmp) {
            fprintf(stderr, "Failed to allocate memory for layers\n");
            return;
        }
        builder->layers = tmp;
        builder->capacity = new_capacity;
    }

    Layer_t prev = builder->layers[builder->size-1];
    size_t input_size = prev->size;
    Layer_t layer = create_layer_from_type(CONV,size,input_size,act,0,0,0,kernel_size);
    if(!layer) {
        fprintf(stderr, "Failed to create layer from type\n");
        return;
    }

    builder->layers[builder->size++] = layer;
}