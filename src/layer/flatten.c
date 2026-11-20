#include "layer.h"
#include "aligned_alloc.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


static size_t align32(size_t n){
    if (n > SIZE_MAX - 31){return 0;}

    return (n + 31) & ~(size_t)31;
}

static void free_layer_flatten(Layer_t layer){
    if (!layer){return;}
    LayerFlatten *flatten = layer->impl;
    if (flatten) {
        aligned_free(flatten->values);
        free(flatten);
    }
    free(layer);
}

static void layer_backward_flatten(
    Layer_t layer,
    const float *output_gradient,
    float *input_gradient
) {
    if (!layer || !output_gradient || !input_gradient) {
        return;
    }

    memcpy(
        input_gradient,
        output_gradient,
        layer->input_size * sizeof(float)
    );
}

static void layer_forward_flatten(Layer_t layer, const float* input) {
    if(!layer){return;}
    LayerFlatten *flatten = layer->impl;
    if (flatten) {

        for(size_t i = 0; i < layer->output_size; ++i) {
            flatten->values[i] = input[i];
        }
    }
}

Layer_t layer_create_flatten(size_t size) {
    if (size == 0) {
        return NULL;
    }
    if (size > SIZE_MAX / sizeof(float))
        goto error;

    Layer_t layer = calloc(1, sizeof(*layer));
    if (!layer){return NULL;}

    LayerFlatten *flatten = calloc(1, sizeof(*flatten));
    if (!flatten) {
        free(layer);
        return NULL;
    }

    layer->size = size;
    layer->input_size = size;
    layer->output_size = size;

    size_t values_size = align32(size * sizeof(float));
    if (!values_size) {goto error;}
    flatten->values = aligned_malloc(values_size,32);
    if (!flatten->values) {goto error;}

    layer->type = FLATTEN;
    layer->impl = flatten;
    layer->trainable = false;
    layer->destroy = free_layer_flatten;
    layer->forward = layer_forward_flatten;
    layer->backward = layer_backward_flatten;
    layer->update = NULL;
    flatten->activation=LINEAR;

    return layer;

    error:
        aligned_free(flatten->values);
    free(flatten);
    free(layer);
    return NULL;

}