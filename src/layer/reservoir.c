
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "vector.h"
#include "aligned_alloc.h"
#include <stdbool.h>
#include "layer.h"
#include <float.h>


static bool checked_add_size(size_t a, size_t b, size_t *result)
{
    if (b > SIZE_MAX - a)
        return false;

    *result = a + b;
    return true;
}

static bool checked_mul_size(size_t a, size_t b, size_t *result)
{
    if (a != 0 && b > SIZE_MAX / a)
        return false;

    *result = a * b;
    return true;
}

static bool align32_checked(size_t n, size_t *result)
{
    if (n > SIZE_MAX - 31)
        return false;

    *result = (n + 31) & ~(size_t)31;
    return true;
}

static float randf_uniform(float lo, float hi) {
    return lo + (hi - lo) * ((float)rand() / (float)RAND_MAX);
}

static int rand_uniform(int lo, int hi){
    return rand() % (hi - lo + 1) + lo;
}

static void *reservoir_aligned_alloc(size_t count, size_t element_size)
{
    size_t bytes;
    size_t aligned_bytes;

    if (!checked_mul_size(count, element_size, &bytes))
        return NULL;

    if (!align32_checked(bytes, &aligned_bytes))
        return NULL;

    /* Évite une allocation de taille nulle. */
    if (aligned_bytes == 0)
        aligned_bytes = 32;

    return aligned_malloc(aligned_bytes, 32);
}

static size_t random_poisson_degree(float lambda, size_t neuron_count)
{
    if (lambda <= 0.0f)
        return 0;

    const float threshold = expf(-lambda);
    float product = 1.0f;
    size_t degree = 0;

    while (product > threshold && degree < neuron_count) {
        product *= randf_uniform(0.0f, 1.0f);
        degree++;
    }

    return degree > 0 ? degree - 1 : 0;
}

static void reservoir_destroy(Layer_t layer)
{
    if (!layer)
        return;

    LayerReservoir *reservoir = layer->impl;

    if (reservoir) {
        aligned_free(reservoir->connexion_start);
        aligned_free(reservoir->connexion);
        aligned_free(reservoir->weights);
        aligned_free(reservoir->input_weights);
        aligned_free(reservoir->state);
        aligned_free(reservoir->next_state);

        free(reservoir);
    }

    free(layer);
}

static float spectral_radius_estimate(
    const LayerReservoir *reservoir,
    size_t iterations)
{
    if (!reservoir || !reservoir->connexion_start ||
        !reservoir->connexion || !reservoir->weights ||
        reservoir->neuron_count == 0 || iterations == 0) {
        return NAN;
    }

    const size_t n = reservoir->neuron_count;

    double *v = malloc(n * sizeof(double));
    double *v_next = malloc(n * sizeof(double));

    if (!v || !v_next) {
        free(v);
        free(v_next);
        return NAN;
    }

    /* Initialisation du vecteur avec une norme non nulle. */
    double initial_norm = 0.0;

    for (size_t i = 0; i < n; ++i) {
        v[i] = (double)randf_uniform(-1.0f, 1.0f);

        if (!isfinite(v[i])) {
            free(v);
            free(v_next);
            return NAN;
        }

        initial_norm += v[i] * v[i];
    }

    initial_norm = sqrt(initial_norm);

    /* Cas improbable : vecteur initial nul. */
    if (!isfinite(initial_norm) || initial_norm < 1e-15) {
        for (size_t i = 0; i < n; ++i) {
            v[i] = 1.0 / sqrt((double)n);
        }
    } else {
        for (size_t i = 0; i < n; ++i) {
            v[i] /= initial_norm;
        }
    }

    double rho = 0.0;

    for (size_t it = 0; it < iterations; ++it) {

        /* v_next = W * v */
        for (size_t i = 0; i < n; ++i) {
            double acc = 0.0;

            const size_t start = reservoir->connexion_start[i];
            const size_t end = reservoir->connexion_start[i + 1];

            for (size_t k = start; k < end; ++k) {
                const size_t source = reservoir->connexion[k];

                if (source >= n || !isfinite(reservoir->weights[k])) {
                    free(v);
                    free(v_next);
                    return NAN;
                }

                acc += (double)reservoir->weights[k] * v[source];
            }

            if (!isfinite(acc)) {
                free(v);
                free(v_next);
                return NAN;
            }

            v_next[i] = acc;
        }

        /* Calcul de la norme du nouveau vecteur. */
        double norm_next = 0.0;

        for (size_t i = 0; i < n; ++i) {
            norm_next += v_next[i] * v_next[i];
        }

        norm_next = sqrt(norm_next);

        if (!isfinite(norm_next)) {
            free(v);
            free(v_next);
            return NAN;
        }

        /* Matrice nulle ou vecteur annulé. */
        if (norm_next < 1e-15) {
            rho = 0.0;
            break;
        }

        rho = norm_next;

        /* Normalisation pour éviter les débordements. */
        for (size_t i = 0; i < n; ++i) {
            v_next[i] /= norm_next;
        }

        double *tmp = v;
        v = v_next;
        v_next = tmp;
    }

    free(v);
    free(v_next);

    if (!isfinite(rho) || rho > FLT_MAX) {
        return NAN;
    }

    return (float)rho;
}

static void reservoir_backward(
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

static void reservoir_forward(Layer_t layer, const float *input)
{
    if (!layer || !layer->impl || !input)
        return;

    LayerReservoir *reservoir = layer->impl;

    const size_t n = reservoir->neuron_count;
    const size_t input_size = layer->input_size;

    const float alpha = reservoir->leak_rate;
    const float beta = 1.0f - alpha;


    for (size_t i = 0; i < n; ++i) {

        float acc = 0.0f;

        /* Connexions récurrentes. */
        const size_t start = reservoir->connexion_start[i];
        const size_t end = reservoir->connexion_start[i + 1];

        for (size_t k = start; k < end; ++k) {
            const size_t source = reservoir->connexion[k];

            acc += reservoir->weights[k] *
                   reservoir->state[source];
        }

        /* Connexions de l'entrée. */
        const float *input_weights =
            reservoir->input_weights + i * input_size;

        acc += dot_product(input_weights, input, input_size);

        reservoir->next_state[i] =
            reservoir->activation.activation(acc);
    }

#ifdef __AVX2__

    const vecf alpha_v = vec_set1(alpha);
    const vecf beta_v = vec_set1(beta);

    size_t i = 0;

    for (; i + VEC_WIDTH <= n; i += VEC_WIDTH) {

        vecf old_state = vec_load(reservoir->state + i);
        vecf new_state = vec_load(reservoir->next_state + i);

        vecf result = vec_add(
            vec_mul(beta_v, old_state),
            vec_mul(alpha_v, new_state)
        );

        vec_store(reservoir->next_state + i, result);
    }

    for (; i < n; ++i) {
        reservoir->next_state[i] =
            beta * reservoir->state[i] +
            alpha * reservoir->next_state[i];
    }

#else

    for (size_t i = 0; i < n; ++i) {
        reservoir->next_state[i] =
            beta * reservoir->state[i] +
            alpha * reservoir->next_state[i];
    }

#endif

    float *tmp = reservoir->state;
    reservoir->state = reservoir->next_state;
    reservoir->next_state = tmp;
}

Layer_t layer_create_reservoir(size_t input_size,
    size_t neuron_count,
    float leak_rate,
    ActivationFunction activation,
    float average_density,
    float spectral_radius) {
    const size_t n = neuron_count;

    Layer_t layer = calloc(1, sizeof(*layer));
    if (!layer){return NULL;}

    LayerReservoir *reservoir = calloc(1, sizeof(*reservoir));

    if (!reservoir) {
        free(layer);
        return NULL;
    }

    reservoir->neuron_count = n;
    reservoir->leak_rate = leak_rate;
    reservoir->activation = activation;
    layer->type = RESERVOIR;
    layer->impl = reservoir;
    layer->size = n;
    layer->input_size = input_size;
    layer->output_size = n;
    layer->trainable = false;
    layer->destroy = reservoir_destroy;
    layer->forward = reservoir_forward;
    layer->backward = reservoir_backward;
    layer->update = NULL;

    reservoir->connexion_start = reservoir_aligned_alloc(n + 1, sizeof(size_t));

    if (!reservoir->connexion_start){goto allocation_error;}

    float lambda = (float)n * average_density;

    if (lambda < 1.0f){lambda = 1.0f;}

    size_t connection_count = 0;
    for (size_t i = 0; i < n; ++i) {
        reservoir->connexion_start[i] = connection_count;
        const size_t degree = random_poisson_degree(lambda, n);

        if (!checked_add_size(connection_count, degree,&connection_count)) {
            goto allocation_error;
        }
    }

    reservoir->connexion_start[n] = connection_count;
    reservoir->connexion = reservoir_aligned_alloc(connection_count, sizeof(size_t));
    reservoir->weights = reservoir_aligned_alloc(connection_count, sizeof(float));
    if (!reservoir->connexion || !reservoir->weights){goto allocation_error;}

    for (size_t i = 0; i < connection_count; ++i) {
        reservoir->connexion[i] =(size_t)(rand() % n);
        reservoir->weights[i] = randf_uniform(-1.0f, 1.0f);
    }

    size_t input_weight_count;
    if (!checked_mul_size(n, input_size, &input_weight_count)){goto allocation_error;}

    reservoir->input_weights = reservoir_aligned_alloc(input_weight_count, sizeof(float));
    if (!reservoir->input_weights){goto allocation_error;}

    for (size_t i = 0; i < input_weight_count; ++i) {
        reservoir->input_weights[i] = randf_uniform(-1.0f, 1.0f);
    }


    reservoir->state = reservoir_aligned_alloc(n, sizeof(float));
    reservoir->next_state = reservoir_aligned_alloc(n, sizeof(float));
    if (!reservoir->state || !reservoir->next_state){goto allocation_error;}
    memset(reservoir->state, 0, n * sizeof(float));
    memset(reservoir->next_state, 0, n * sizeof(float));

    const float rho = spectral_radius_estimate(reservoir, 50);
    if (!isfinite(rho)){goto allocation_error;}
    if (rho > 1e-12f) {
        const float scale = spectral_radius / rho;
        for (size_t i = 0; i < connection_count; ++i) {
            reservoir->weights[i] *= scale;
        }
    }

    return layer;

allocation_error:
    fprintf(stderr, "Erreur lors de la création du réservoir\n");
    reservoir_destroy(layer);
    return NULL;
}