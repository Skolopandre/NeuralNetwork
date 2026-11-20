
#ifndef LAYER_H
#define LAYER_H

#include <stdbool.h>
#include <stddef.h>
#include "activation.h"

/* Forward declarations */
typedef struct Layer Layer;
typedef Layer *Layer_t;

typedef struct LayerConv{
    size_t size;
    size_t input_size;
    size_t stride;

    size_t kernel_width;
    size_t kernel_height;
    size_t padding_x;
    size_t padding_y;

    void* memory;

    float* restrict values;
    float* restrict input_values;
    float* restrict weights;
    float* restrict bias;
    float* restrict deltas;

    float* restrict moment;
    float* restrict variance;
    float* restrict moment_bias;
    float* restrict variance_bias;

    ActivationFunction activation;
}LayerConv;

typedef struct LayerDense {
    size_t size;
    size_t input_size;
    size_t weight_stride;

    void *memory;

    float *restrict values;       // [size]
    float *restrict input_values; // [input_size]
    float *restrict weights;      // [size * weight_stride]
    float *restrict bias;         // [size]
    float *restrict deltas;       // [size]

    float *restrict moment;       // [size * weight_stride]
    float *restrict variance;     // [size * weight_stride]
    float *restrict moment_bias;  // [size]
    float *restrict variance_bias;// [size]

    ActivationFunction activation;
}LayerDense;

typedef struct LayerReservoir{
    size_t neuron_count;

    size_t *restrict connexion_start;
    size_t *restrict connexion;

    float *restrict weights;
    float *restrict input_weights;

    float *restrict state;
    float *restrict next_state;

    float leak_rate;
    ActivationFunction activation;
} LayerReservoir;

typedef struct LayerFlatten {
    float* restrict values;
    ActivationFunction activation;
}LayerFlatten;

/* Types of layer implementations */
typedef enum {
    DENSE,
    RESERVOIR,
    FLATTEN,
    CONV
} LayerType;

/* Generic layer interface */
struct Layer {
    LayerType type;
    size_t size;
    size_t input_size;
    size_t output_size;

    void *impl;
    bool trainable;

    void (*destroy)(Layer_t layer);
    void (*forward)(Layer_t layer,const float *input);
    void (*backward)(Layer_t layer,const float *output_gradient,float *input_gradient);
    void (*update)(Layer_t layer);
};

/* Generic interface functions */
void layer_destroy(Layer_t layer);

void layer_forward(Layer_t layer,const float *input);

void layer_backward(Layer_t layer,const float *output_gradient,float *input_gradient);

void layer_update(Layer_t layer);

Layer_t create_layer_from_type(LayerType type,size_t size,size_t input_size,ActivationFunction act,float leak_rate, float average_density,float spectral_radius,size_t* kernel_size);
Layer_t layer_create_dense(size_t size,size_t input_size,ActivationFunction act);
Layer_t layer_create_reservoir(size_t input,size_t neuron_count, float leak_rate,ActivationFunction activation,float average_density,float spectral_radius);
Layer_t layer_create_flatten(size_t size);
Layer_t layer_create_conv(size_t size,size_t input_size,ActivationFunction act,size_t* kernel_size);

size_t layer_get_size(Layer_t l);
size_t layer_get_input_size(Layer_t l);
size_t layer_get_output_size(Layer_t l);
float* layer_get_deltas(Layer_t l);
float* layer_get_values(Layer_t l);
ActivationFunction layer_get_activation(Layer_t l);
float* layer_get_weights(Layer_t l);
float* layer_get_bias(Layer_t l);
float* layer_get_moment(Layer_t l);
float* layer_get_moment_bias(Layer_t l);
float* layer_get_variance(Layer_t l);
float* layer_get_variance_bias(Layer_t l);
size_t layer_get_weight_stride(Layer_t l);
#endif