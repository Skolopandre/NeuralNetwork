#include <stdlib.h>
#include <math.h>
#include "optimizer.h"
#include "network.h"
#include "vector.h"
#include "layer.h"


static void adamw_set_learning_rate(Optimizer_t opt,float learning_rate){
    AdamWContext *ctx = opt->context;
    ctx->learning_rate = learning_rate;
}

/*
m = β1·m + (1−β1)·g
v = β2·v + (1−β2)·g²

m̂ = m/(1−β1ᵗ)
v̂ = v/(1−β2ᵗ)

w = w − α·m̂/(√v̂+ε)
w = w − α·λ·w
*/
static void adamw_update_layer(Optimizer *opt, Layer *layer, const float *prev_values)
{
    AdamWContext *ctx = opt->context;
    size_t size = layer_get_size(layer);
    size_t input_size = layer_get_input_size(layer);
    size_t stride = layer_get_weight_stride(layer);
    float *weights = layer_get_weights(layer);
    float *bias = layer_get_bias(layer);
    float *deltas = layer_get_deltas(layer);
    float *moment = layer_get_moment(layer);
    float *variance = layer_get_variance(layer);
    float *moment_bias = layer_get_moment_bias(layer);
    float *variance_bias = layer_get_variance_bias(layer);

    float beta1_pow = powf(ctx->beta1, ctx->step +1);
    float beta2_pow = powf(ctx->beta2, ctx->step +1);
    float bc1 = 1.0f - beta1_pow;
    float bc2 = 1.0f - beta2_pow;

    const vecf v_lr = vec_set1(ctx->learning_rate);
    const vecf v_beta1 = vec_set1(ctx->beta1);
    const vecf v_beta2 = vec_set1(ctx->beta2);
    const vecf v_eps = vec_set1(ctx->epsilon);
    const vecf v_wd = vec_set1(ctx->weight_decay);
    const vecf v_one = vec_set1(1.0f);
    const vecf v_inv_bc1 = vec_set1(1.0f / bc1);
    const vecf v_inv_bc2 = vec_set1(1.0f / bc2);
    const vecf v_omb1 = vec_sub(v_one, v_beta1);  // (1 - beta1)
    const vecf v_omb2 = vec_sub(v_one, v_beta2);  // (1 - beta2)

    for (size_t out = 0; out < size; ++out) {
        const float delta = deltas[out];
        const vecf v_delta = vec_set1(delta);

        size_t in = 0;
       const size_t base = out * stride;;

        for (; in + VEC_WIDTH <= input_size; in += VEC_WIDTH) {
            const size_t idx = base + in;

            vecf w = vec_load(&weights[idx]);
            vecf m = vec_load(&moment[idx]);
            vecf v = vec_load(&variance[idx]);
            vecf x = vec_load(&prev_values[in]);

            //g = delta * x
            vecf g = vec_mul(v_delta, x);

            //m = beta1 * m + (1 - beta1) * g
            m = vec_add(vec_mul(v_beta1, m), vec_mul(v_omb1, g));

            //v = beta2 * v + (1 - beta2) * g²
            vecf g2 = vec_mul(g, g);
            v = vec_add(vec_mul(v_beta2, v), vec_mul(v_omb2, g2));

            // m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
            vecf m_hat = vec_mul(m, v_inv_bc1);
            vecf v_hat = vec_mul(v, v_inv_bc2);

            //w -= lr * (m_hat / (sqrt(v_hat) + eps))
            vecf denom = vec_add(vec_sqrt(v_hat), v_eps);
            vecf step = vec_mul(v_lr, vec_div(m_hat, denom));
            w = vec_sub(w, step);

            //w -= lr * weight_decay * w
            w = vec_sub(w, vec_mul(vec_mul(v_lr, v_wd), w));

            vec_store(&weights[idx], w);
            vec_store(&moment[idx], m);
            vec_store(&variance[idx], v);
        }

        for (; in < input_size; in++) {
            const size_t idx = base + in;
            const float g = delta * prev_values[in];
            float m = moment[idx] = ctx->beta1 * moment[idx] + (1.0f - ctx->beta1) * g;
            float v = variance[idx] = ctx->beta2 * variance[idx] + (1.0f - ctx->beta2) * g * g;
            float m_hat = m / bc1;
            float v_hat = v / bc2;
            weights[idx] -= ctx->learning_rate * (m_hat / (sqrtf(v_hat) + ctx->epsilon));
            weights[idx] -= ctx->learning_rate * ctx->weight_decay * weights[idx];
        }

        float g_b = delta;
        float mb = moment_bias[out] = ctx->beta1 * moment_bias[out] + (1.0f - ctx->beta1) * g_b;
        float vb = variance_bias[out] = ctx->beta2 * variance_bias[out] + (1.0f - ctx->beta2) * g_b * g_b;
        float mb_hat = mb / bc1;
        float vb_hat = vb / bc2;
        bias[out] -= ctx->learning_rate * mb_hat / (sqrtf(vb_hat) + ctx->epsilon);
    }
}

Optimizer_t optimizer_create_adamw(float learning_rate, float beta1, float beta2, float eps, float wd)
{
    Optimizer_t opt = malloc(sizeof(Optimizer));
    if (!opt) {
        perror("Erreur d'allocation de l'optimiseur AdamW");
        return NULL;
    }

    AdamWContext *ctx = malloc(sizeof(*ctx));
    if (!ctx) {
        perror("Erreur de création du contexte AdamW");
        free(opt);
        return NULL;
    }

    ctx->learning_rate = learning_rate;
    ctx->beta1 = beta1;
    ctx->beta2 = beta2;
    ctx->epsilon = eps;
    ctx->weight_decay = wd;
    ctx->step = 0;

    opt->type = ADAMW;
    opt->update_layer = adamw_update_layer;
    opt->context = ctx;
    opt->set_learning_rate=adamw_set_learning_rate;
    return opt;
}


void step_Adamw(Optimizer *opt) {
    AdamWContext *ctx = ((AdamWContext *)opt->context);
    ctx->step++;
}