#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <float.h>
#include <string.h>
#include "network.h"
#include <stdlib.h>
#include "optimizer.h"
#include "aligned_alloc.h"


struct Network {
    size_t num_layers;
    Layer_t *layers;

    size_t input_size;
    size_t output_size;

    Optimizer_t optimizer;

    /* Buffers temporaires pour la rétropropagation */
    float *gradient_a;
    float *gradient_b;
    size_t gradient_capacity;
};


float mse_loss(Network_t NN, const float *target){
    if (!NN || !target || NN->num_layers == 0){return NAN;}

    const float *output = layer_get_values(NN->layers[NN->num_layers - 1]);
    if (!output || NN->output_size == 0){return NAN;}

    float loss = 0.0f;
    for (size_t i = 0; i < NN->output_size; ++i) {
        const float error = target[i] - output[i];
        loss += error * error;
    }

    return loss / (float)NN->output_size;
}

static size_t align32(size_t n)
{
    return (n + 31) & ~(size_t)31;
}

Network_t init_network(network_builder_t builder, OptimizerType optim_type){
    if (!builder || builder->size == 0) {
        fprintf(stderr, "Invalid network builder\n");
        return NULL;
    }

    // Vérification de la cohérence dimensionnelle
    for (size_t i = 1; i < builder->size; i++) {
        if (builder->layers[i]->input_size !=
            builder->layers[i - 1]->output_size) {

            fprintf(stderr, "Incompatible layer dimensions at index %zu\n", i);
            return NULL;
        }
    }

    Network_t network = calloc(1, sizeof(struct Network));

    if (!network) {
        fprintf(stderr, "Failed to allocate network\n");
        return NULL;
    }

    network->num_layers = builder->size;
    network->layers = builder->layers;

    network->input_size = builder->layers[0]->input_size;
    network->output_size = builder->layers[builder->size - 1]->output_size;

    // Calcul de la dimension maximale des gradients
    network->gradient_capacity = network->input_size;

    for (size_t i = 0; i < network->num_layers; i++) {
        Layer_t layer = network->layers[i];

        if (layer->input_size > network->gradient_capacity)
            network->gradient_capacity = layer->input_size;

        if (layer->output_size > network->gradient_capacity)
            network->gradient_capacity = layer->output_size;
    }

    // Allocation des buffers de gradients
    size_t gradient_bytes = network->gradient_capacity * sizeof(float);

    network->gradient_a = aligned_malloc(gradient_bytes, 32);
    network->gradient_b = aligned_malloc(gradient_bytes, 32);

    if (!network->gradient_a || !network->gradient_b) {
        fprintf(stderr, "Failed to allocate gradient buffers\n");

        aligned_free(network->gradient_a);
        aligned_free(network->gradient_b);
        free(network);

        return NULL;
    }

    // Création de l'optimiseur
    network->optimizer = create_optimizer_from_type(optim_type);

    if (!network->optimizer) {
        fprintf(stderr, "Failed to create optimizer\n");

        aligned_free(network->gradient_a);
        aligned_free(network->gradient_b);
        free(network);

        return NULL;
    }

    // Transfert de propriété des couches au réseau
    builder->layers = NULL;
    builder->size = 0;
    builder->capacity = 0;

    return network;

}


void free_network(Network_t NN)
{
    if (!NN)
        return;

    for (size_t i = 0; i < NN->num_layers; ++i) {
        layer_destroy(NN->layers[i]);
    }

    free(NN->layers);

    aligned_free(NN->gradient_a);
    aligned_free(NN->gradient_b);

    free_optimizer(NN->optimizer);

    free(NN);
}

void forward_propagation(Network_t NN, const float *input){
    if (!NN || !input || NN->num_layers == 0){return ;}

    const float *current_input = input;
    for (size_t i = 0; i < NN->num_layers; ++i) {

        Layer_t current = NN->layers[i];
        if (!current){return ;}

        layer_forward(current, current_input);

        current_input = layer_get_values(current);
        if (!current_input){return ;}
    }
}

void backpropagation(Network_t NN,const float* restrict input, const float *restrict target){
    if (!NN || !target || NN->num_layers == 0){return ;}

    const size_t L = NN->num_layers;
    const size_t output_size = NN->output_size;

    if (!NN->gradient_a || !NN->gradient_b){return ;}

    Layer_t output_layer = NN->layers[L - 1];
    const float *output = layer_get_values(output_layer);

    if (!output){return ;}

    float *gradient_out = NN->gradient_a;

    for (size_t i = 0; i < output_size; ++i) {
        gradient_out[i] =
            2.0f * (output[i] - target[i]) /
            (float)output_size;
    }

    float *gradient_in = NN->gradient_b;

    for (size_t l = L; l-- > 0;) {
        Layer_t current = NN->layers[l];

        if (!current->backward){return ;}
        layer_backward(current,gradient_out,gradient_in);
        float *tmp = gradient_out;
        gradient_out = gradient_in;
        gradient_in = tmp;
    }

    if (NN->optimizer) {
        for (size_t l = 0; l < L; ++l) {
            Layer_t current = NN->layers[l];
            if (!current->trainable){continue;}
            if (!NN->optimizer->update_layer){return ;}

            const float *previous_values = (l == 0)? input : layer_get_values(NN->layers[l - 1]);
            if (!previous_values){return;}

            NN->optimizer->update_layer(NN->optimizer,current,previous_values);
        }
    }

    if (NN->optimizer && NN->optimizer->type == ADAMW) {
        step_Adamw(NN->optimizer);
    }
}

void network_reset_reservoir(Network_t NN) {
    for(size_t i = 0; i < NN->num_layers; ++i) {
        if (NN->layers[i]->type==RESERVOIR && NN->layers[i]->impl) {
            LayerReservoir *reservoir = NN->layers[i]->impl;

            memset(
            reservoir->state,
            0,
            reservoir->neuron_count * sizeof(float));

            memset(
            reservoir->next_state,
            0,
            reservoir->neuron_count * sizeof(float));
        }
    }
}


size_t predict(Network_t NN)
{

    const float *output =
        layer_get_values(NN->layers[NN->num_layers - 1]);

    if (!output)
        return SIZE_MAX;

    size_t best_index = 0;
    float best_value = -FLT_MAX;

    for (size_t i = 0; i < NN->output_size; ++i) {
        if (output[i] > best_value) {
            best_index = i;
            best_value = output[i];
        }
    }

    return best_index;

}


void predict_full(Network_t NN, const float *input, float *output)
{
    forward_propagation(NN, input);

    Layer *out = NN->layers[NN->num_layers - 1];
    memcpy(output, layer_get_values(out), out->size * sizeof(float));
}

void save_weights(Network_t NN, const char *filename)
{
    FILE *f = fopen(filename, "w");
    if (!f){
        perror("fopen");
        return;
    }

    for (int l = 1; l < NN->num_layers; l++){
        Layer_t cur = NN->layers[l];
        float* weights = layer_get_values(cur);
        float* bias = layer_get_bias(cur);
        for (int out = 0; out < cur->size; out++){
            for (int in = 0; in < cur->input_size; in++){
                fprintf(f, "%f ", weights[out * cur->input_size + in]);
            }
            fprintf(f, "%f\n", bias[out]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void load_weights(Network_t NN, const char *filename)
{
    FILE *f = fopen(filename, "r");
    if (!f){
        perror("fopen");
        return;
    }

    for (int l = 1; l < NN->num_layers; l++){
        Layer_t cur = NN->layers[l];
        float* weights = layer_get_values(cur);
        float* bias = layer_get_bias(cur);

        for (int out = 0; out < cur->size; out++){
            for (int in = 0; in < cur->input_size; in++){
                int idx = out * cur->input_size + in;
                if (fscanf(f, "%f", &weights[idx]) != 1){
                    fprintf(stderr, "Erreur de lecture de %s\n", filename);
                    fclose(f);
                    return;
                }
            }

            if (fscanf(f, "%f", &bias[out]) != 1){
                fprintf(stderr, "Erreur de lecture de %s\n", filename);
                fclose(f);
                return;
            }
        }
    }

    fclose(f);

    printf("Poids et biais chargés depuis %s\n", filename);
}

size_t network_get_num_layers(Network_t NN)
{
    return NN ? NN->num_layers : 0;
}


Layer_t network_get_layer(Network_t NN, size_t index)
{
    if (!NN || index >= NN->num_layers)
        return NULL;

    return NN->layers[index];
}


Optimizer_t network_get_optimizer(Network_t NN)
{
    return NN ? NN->optimizer : NULL;
}


size_t network_get_input_size(Network_t NN)
{
    return NN ? NN->input_size : 0;
}


size_t network_get_output_size(Network_t NN)
{
    return NN ? NN->output_size : 0;
}

void network_set_num_layers(Network_t NN, size_t num_layers) {
    NN->num_layers = num_layers;
}
void network_set_layers(Network_t NN,Layer_t* layers) {
    NN->layers = layers;
}
void network_set_input_size(Network_t NN, size_t input_size) {
    NN->input_size = input_size;
}
void network_set_output_size(Network_t NN, size_t output_size) {
    NN->output_size = output_size;
}

void network_set_opt(Network_t NN, Optimizer_t optimizer) {
    NN->optimizer = optimizer;
}

void network_set_grad_a(Network_t NN, void* ptr) {
    NN->gradient_a = ptr;
}
void network_set_grad_b(Network_t NN, void* ptr) {
    NN->gradient_b = ptr;
}
void network_set_gradient_capacity(Network_t NN, size_t capacity) {
    NN->gradient_capacity=capacity;
}