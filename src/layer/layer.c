#include "layer.h"
#include "activation.h"
#include <stdio.h>


Layer_t create_layer_from_type(LayerType type,
    size_t size,
    size_t input_size,
    ActivationFunction act,
    float leak_rate,
    float average_density,
    float spectral_radius,
    size_t* kernel_size
)          //on surcharge en paramètre. Ce ne sera de toute façon qu'appelé en interne par le builder
{
    switch (type)
    {
    case DENSE:
        return layer_create_dense(size, input_size, act);

    case RESERVOIR:
        return layer_create_reservoir(input_size, size, leak_rate,act, average_density,spectral_radius);

    case FLATTEN:
        return layer_create_flatten(size);

    case CONV:
        return layer_create_conv(size,input_size,act,kernel_size);

    default:
        fprintf(stderr, "layer DENSE par défaut\n");
        return layer_create_dense(size, input_size, act);
    }
}

void layer_destroy(Layer_t layer) {
    layer->destroy(layer);
}

void layer_forward(Layer_t l,const float *input) {
    l->forward(l, input);
}

void layer_backward(Layer_t l,const float *output_gradient,float *input_gradient) {
    l->backward(l, output_gradient, input_gradient);
}

void layer_update(Layer_t layer) {
    layer->update(layer);
}

size_t layer_get_size(Layer_t l) {
    return l->size;
}
size_t layer_get_input_size(Layer_t l) {
    return l->input_size;
}
size_t layer_get_output_size(Layer_t l) {
    return l->output_size;
}
float* layer_get_deltas(Layer_t l){
    switch(l->type) {
        case DENSE:
            LayerDense *dense = l->impl;
            return dense->deltas;
        case RESERVOIR:
            return NULL;
        case FLATTEN:
            return NULL;
        case CONV :
            LayerConv *conv = l->impl;
            return conv->deltas;
    }
}
float* layer_get_values(Layer_t l) {
    switch(l->type) {
        case DENSE:
            LayerDense *dense = l->impl;
            return dense->values;
        case RESERVOIR:
            LayerReservoir *reservoir = l->impl;
            return reservoir->state;
        case FLATTEN:
            LayerFlatten *flatten = l->impl;
            return flatten->values;
        case CONV:
            LayerConv *conv = l->impl;
            return conv->values;
    }
}
ActivationFunction layer_get_activation(Layer_t l) {
    switch(l->type) {
        case DENSE:
            LayerDense *dense = l->impl;
        return dense->activation;
        case RESERVOIR:
            LayerReservoir *reservoir = l->impl;
        return reservoir->activation;
        case FLATTEN:
            LayerFlatten *flat = l->impl ;
            return flat->activation;
        case CONV:
            LayerConv *conv = l->impl;
            return conv->activation;
    }
}
float* layer_get_weights(Layer_t l) {
    switch(l->type) {
        case DENSE:
            LayerDense *dense = l->impl;
        return dense->weights;
        case RESERVOIR:
            LayerReservoir *reservoir = l->impl;
        return reservoir->weights;
        case FLATTEN:
            return NULL;
        case CONV:
            LayerConv *conv = l->impl;
            return conv->weights;
    }
}
float* layer_get_bias(Layer_t l){
    switch(l->type) {
        case DENSE:
            LayerDense *dense = l->impl;
        return dense->bias;
        case RESERVOIR:
            return NULL;
        case FLATTEN:
            return NULL;
        case CONV:
            LayerConv *conv = l->impl;
        return conv->bias;
    }
}
float* layer_get_moment(Layer_t l){
    switch(l->type) {
        case DENSE:
            LayerDense *dense = l->impl;
        return dense->moment;
        case RESERVOIR:
            return NULL;
        case FLATTEN:
            return NULL;
        case CONV:
            LayerConv *conv = l->impl;
        return conv->moment;
    }
}
float* layer_get_moment_bias(Layer_t l) {
        switch(l->type) {
            case DENSE:
                LayerDense *dense = l->impl;
            return dense->moment_bias;
            case RESERVOIR:
                return NULL;
            case FLATTEN:
                return NULL;
            case CONV:
                LayerConv *conv = l->impl;
            return conv->moment_bias;

        }
}
float* layer_get_variance(Layer_t l){
    switch(l->type) {
        case DENSE:
            LayerDense *dense = l->impl;
        return dense->variance;
        case RESERVOIR:
            return NULL;
        case FLATTEN:
            return NULL;
        case CONV:
            LayerConv *conv = l->impl;
        return conv->variance;
    }
}
float* layer_get_variance_bias(Layer_t l){
    switch(l->type) {
        case DENSE:
            LayerDense *dense = l->impl;
        return dense->variance_bias;
        case RESERVOIR:
            return NULL;
        case FLATTEN:
            return NULL;
        case CONV:
            LayerConv *conv = l->impl;
        return conv->variance_bias;
    }
}
size_t layer_get_weight_stride(Layer_t l){
    switch(l->type) {
        case DENSE:
            LayerDense *dense = l->impl;
        return dense->weight_stride;
        case RESERVOIR:
            return NULL;
        case FLATTEN:
            return NULL;
        case CONV:
            LayerConv *conv = l->impl;
        return conv->stride;
    }
}