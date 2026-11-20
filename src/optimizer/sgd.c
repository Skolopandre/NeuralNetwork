
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "optimizer.h"
#include "network.h"
#include "vector.h"


static void sgd_set_learning_rate(Optimizer_t opt,float learning_rate){
    SGDContext *ctx = opt->context;
    ctx->learning_rate = learning_rate;
}

static void sgd_update_layer(Optimizer *opt, Layer *layer, const float *prev_values)
{
    SGDContext *ctx = opt->context;
    size_t size = layer_get_size(layer);
    size_t input_size = layer_get_input_size(layer);
    size_t stride = layer_get_weight_stride(layer);
    float *weights = layer_get_weights(layer);
    float *bias = layer_get_bias(layer);
    float *deltas = layer_get_deltas(layer);

    // --- CONSTANTES VECTORIELLES ---
    const vecf v_lr = vec_set1(ctx->learning_rate);

    for (size_t out = 0; out < size; ++out) {
        const float delta = deltas[out];
        const vecf v_delta = vec_set1(delta);

        size_t in = 0;
        const size_t base = out * stride;;

        for (; in + VEC_WIDTH <= input_size; in += VEC_WIDTH) {
            const size_t idx = base + in;
            vecf w = vec_load(&weights[idx]);
            vecf x = vec_load(&prev_values[in]);
            vecf grad = vec_mul(v_delta, x);        // grad = delta * x
            w = vec_sub(w, vec_mul(v_lr, grad));    // w -= lr * grad
            vec_store(&weights[idx], w);
        }

        for (; in < input_size; in++) {
            size_t idx = base + in;
            weights[idx] -= ctx->learning_rate * delta * prev_values[in];
        }

        bias[out] -= ctx->learning_rate * delta;
    }
}

Optimizer_t optimizer_create_sgd(float learning_rate)
{
    Optimizer_t opt = malloc(sizeof(Optimizer));
    if (!opt) {
        perror("Erreur d'allocation de l'optimiseur SGD");
        return NULL;
    }

    SGDContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        perror("Erreur de création du contexte SGD");
        free(opt);
        return NULL;
    }
    ctx->learning_rate = learning_rate;
    opt->type = SGD;
    opt->update_layer = sgd_update_layer;
    opt->context = ctx;


    return opt;
}