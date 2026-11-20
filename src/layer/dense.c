#include "layer.h"
#include "aligned_alloc.h"
#include "vector.h"
#include <math.h>
#include <stdlib.h>
#include "activation.h"
#include <string.h>

static size_t align32(size_t n){
    if (n > SIZE_MAX - 31){return 0;}

    return (n + 31) & ~(size_t)31;
}

static void backward_layer_dense(Layer_t layer,const float *output_gradient,float *input_gradient){
    LayerDense *dense = layer->impl;

    if (!dense || !output_gradient || !input_gradient){return;}

    const size_t input_size = dense->input_size;
    const size_t output_size = dense->size;

    const float *restrict weights = dense->weights;
    float *restrict deltas = dense->deltas;

    for (size_t out = 0; out < output_size; ++out) {
        deltas[out] =
            output_gradient[out] *
            dense->activation.derivative(dense->values[out]);
    }

    size_t in = 0;

    for (; in + VEC_WIDTH <= input_size; in += VEC_WIDTH) {
        vecf acc = vec_set1(0.0f);

        for (size_t out = 0; out < output_size; ++out) {
            const float *w =
                weights + out * dense->weight_stride + in;

            vecf wv = vec_load(w);
            vecf dv = vec_set1(deltas[out]);

            acc = vec_fma(wv, dv, acc);
        }

        vec_store(input_gradient + in, acc);
    }

    //reste scalaire
    for (; in < input_size; ++in) {
        float sum = 0.0f;

        for (size_t out = 0; out < output_size; ++out) {
            sum +=
                weights[out * dense->weight_stride + in] *
                deltas[out];
        }

        input_gradient[in] = sum;
    }
}

static void forward_layer_dense(Layer_t layer,const float *input){
    LayerDense *dense = layer->impl;

    if (!dense || !input)
        return;

    memcpy(
        dense->input_values,
        input,
        dense->input_size * sizeof(float)
    );

    for (size_t out = 0; out < dense->size; ++out) {
        const float *w =
            dense->weights + out * dense->weight_stride;

        float sum = dense->bias[out];

        sum += dot_product(w, input, dense->input_size);

        dense->values[out] = dense->activation.activation(sum);
    }
}

static void free_layer_dense(Layer_t layer){
    if (!layer){return;}
    LayerDense *dense = layer->impl;
    if (dense) {
        aligned_free(dense->memory);
        free(dense);
    }
    free(layer);
}


Layer_t layer_create_dense(size_t size,size_t input_size,ActivationFunction act){
    if (size == 0 || input_size == 0 ||
        !act.activation || !act.derivative) {
        return NULL;
    }

    Layer_t layer = calloc(1, sizeof(*layer));
    if (!layer){return NULL;}

    LayerDense *dense = calloc(1, sizeof(*dense));
    if (!dense) {
        free(layer);
        return NULL;
    }

    layer->size = size;
    layer->input_size = input_size;
    layer->output_size = size;
    dense->activation = act;
    dense->size = size;
    dense->input_size = input_size;

    size_t weight_row_bytes = align32(input_size * sizeof(float));
    if (!weight_row_bytes){goto error;}

    dense->weight_stride = weight_row_bytes / sizeof(float);

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

    dense->memory = aligned_malloc(total, 32);
    if (!dense->memory)
        goto error;

    memset(dense->memory, 0, total);

    char *ptr = dense->memory;

    dense->values = (float *)ptr;
    ptr += values_bytes;

    dense->input_values = (float *)ptr;
    ptr += input_values_bytes;

    dense->weights = (float *)ptr;
    ptr += weights_bytes;

    dense->bias = (float *)ptr;
    ptr += bias_bytes;

    dense->deltas = (float *)ptr;
    ptr += deltas_bytes;

    dense->moment = (float *)ptr;
    ptr += moment_bytes;

    dense->variance = (float *)ptr;
    ptr += variance_bytes;

    dense->moment_bias = (float *)ptr;
    ptr += moment_bias_bytes;

    dense->variance_bias = (float *)ptr;

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

            dense->weights[out * dense->weight_stride + in] =
                (2.0f * r - 1.0f) * limit;
        }
    }

    layer->type = DENSE;
    layer->impl = dense;
    layer->trainable = true;

    layer->destroy = free_layer_dense;
    layer->forward = forward_layer_dense;
    layer->backward = backward_layer_dense;
    layer->update = NULL; // À remplacer par dense_update.

    return layer;

error:
    aligned_free(dense->memory);
    free(dense);
    free(layer);
    return NULL;
}