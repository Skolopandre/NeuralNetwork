
#include "layer.h"
#include "aligned_alloc.h"
#include <math.h>
#include <stdlib.h>
#include "activation.h"
#include <string.h>

static size_t align32(size_t n){
    if (n > SIZE_MAX - 31){return 0;}

    return (n + 31) & ~(size_t)31;
}

static void backward_layer_conv(Layer_t layer,const float *output_gradient,float *input_gradient){

}

static void forward_layer_conv(Layer_t layer,const float *input){

}

static void free_layer_conv(Layer_t layer){
    if (!layer){return;}
    LayerDense *conv = layer->impl;
    if (conv) {
        aligned_free(conv->memory);
        free(conv);
    }
    free(layer);
}


Layer_t layer_create_conv(size_t size,size_t input_size,ActivationFunction act,size_t* kernel_size){
    if (size == 0 || input_size == 0 ||
        !act.activation || !act.derivative) {
        return NULL;
    }

    Layer_t layer = calloc(1, sizeof(*layer));
    if (!layer){return NULL;}

    LayerConv *conv = calloc(1, sizeof(*conv));
    if (!conv) {
        free(layer);
        return NULL;
    }

    layer->size = size;
    layer->input_size = input_size;
    layer->output_size = size;
    conv->activation = act;
    conv->size = size;
    conv->input_size = input_size;
    conv->kernel_width = kernel_size[0];
    conv->kernel_height = kernel_size[1];
    conv->padding_y = (conv->kernel_width - 1)/2;
    conv->padding_x = (conv->kernel_height - 1)/2;

    size_t weight_row_bytes = align32(input_size * sizeof(float));
    if (!weight_row_bytes){goto error;}

    conv->stride = weight_row_bytes / sizeof(float);

    size_t values_bytes = align32(size * sizeof(float));
    size_t input_values_bytes = align32(input_size * sizeof(float));
    size_t weights_bytes = align32(size * weight_row_bytes);
    size_t bias_bytes = align32(size * sizeof(float));
    size_t deltas_bytes = align32(size * sizeof(float));

    size_t moment_bytes = weights_bytes;
    size_t variance_bytes = weights_bytes;
    size_t moment_bias_bytes = bias_bytes;
    size_t variance_bias_bytes = bias_bytes;

    if (!values_bytes || !input_values_bytes ||
        !weights_bytes || !bias_bytes || !deltas_bytes)
        goto error;

    size_t total = 0;
    size_t sizes[] = {
        values_bytes,
        input_values_bytes,
        weights_bytes,
        bias_bytes,
        deltas_bytes,
        moment_bytes,
        variance_bytes,
        moment_bias_bytes,
        variance_bias_bytes
    };

    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); ++i) {
        if (total > SIZE_MAX - sizes[i])
            goto error;

        total += sizes[i];
    }

    conv->memory = aligned_malloc(total, 32);
    if (!conv->memory)
        goto error;

    memset(conv->memory, 0, total);

    char *ptr = conv->memory;

    conv->values = (float *)ptr;
    ptr += values_bytes;

    conv->input_values = (float *)ptr;
    ptr += input_values_bytes;

    conv->weights = (float *)ptr;
    ptr += weights_bytes;

    conv->bias = (float *)ptr;
    ptr += bias_bytes;

    conv->deltas = (float *)ptr;
    ptr += deltas_bytes;

    conv->moment = (float *)ptr;
    ptr += moment_bytes;

    conv->variance = (float *)ptr;
    ptr += variance_bytes;

    conv->moment_bias = (float *)ptr;
    ptr += moment_bias_bytes;

    conv->variance_bias = (float *)ptr;

    float limit;

    if (act.activation == relu_activation ||
        act.activation == lrelu_activation) {
        limit = sqrtf(2.0f / (float)input_size);
    } else {
        limit = sqrtf(6.0f / (float)(input_size + size));
    }

    for (size_t out = 0; out < size; ++out) {
        for (size_t in = 0; in < input_size; ++in) {
            float r = (float)rand() / (float)RAND_MAX;

            conv->weights[out * conv->stride + in] =
                (2.0f * r - 1.0f) * limit;
        }
    }

    layer->type = CONV;
    layer->impl = conv;
    layer->trainable = true;

    layer->destroy = free_layer_conv;
    layer->forward = forward_layer_conv;
    layer->backward = backward_layer_conv;
    layer->update = NULL; // À remplacer par conv_update.

    return layer;

error:
    aligned_free(conv->memory);
    free(conv);
    free(layer);
    return NULL;
}