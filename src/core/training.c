#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <float.h>
#include "Training.h"
#include "layer.h"

typedef struct TrainingController {
    float *history;               // Historique des losses
    size_t window_size;              // Taille de la fenêtre glissante
    size_t count;                    // Nombre de losses stockées
    size_t index;                    // Index circulaire
    float sum;                    // Somme des losses dans la fenêtre
    float target_loss;            // Seuil d'arrêt (optionnel, peut être ignoré)

    float min_loss;               // Meilleure loss vue jusqu'ici
    size_t patience;                 // Nombre d'itérations sans amélioration avant arrêt
    size_t no_improvement_count;     // Compteur d'itérations sans amélioration
    float tolerance;              // Tolérance : variation minimale significative (ex: 0.0001)
    size_t context;                    //context d'arrêt
} TrainingController;

static void print_context(int context) {
    switch (context) {
        case 1:
            printf("Target loss achieved\n");
        return;
        case 2:
            printf("Patience achieved\n");
        return;
    }
}

TrainingController_t create_training_controller(
size_t window_size,
float target_loss,
size_t patience,          // Nombre d'itérations sans amélioration
float tolerance        // Tolérance pour considérer qu'il n'y a pas d'amélioration
) {
    TrainingController_t c = malloc(sizeof(TrainingController));
    if (!c) {
        perror("Erreur d'allocation du TrainingController");
        return NULL;
    }

    c->history = calloc(window_size, sizeof(float));
    if (!c->history) {
        perror("Erreur d'allocation de l'historique");
        free(c);
        return NULL;
    }

    c->window_size = window_size;
    c->count = 0;
    c->index = 0;
    c->sum = 0.0f;
    c->target_loss = target_loss;

    c->min_loss = FLT_MAX;
    c->patience = patience;
    c->no_improvement_count = 0;
    c->tolerance = tolerance;

    return c;
}

void free_training_controller(TrainingController_t c){
    if(!c){return;}

    free(c->history);
    free(c);
}

void training_controller_reset(TrainingController_t controller) {
    memset(controller->history, 0, controller->window_size * sizeof(float));
    controller->count = 0;
    controller->index = 0;
    controller->sum = 0.0f;

    controller->min_loss = FLT_MAX;
    controller->no_improvement_count = 0;
}

void training_controller_update(TrainingController_t c, float loss) {
    if (c->count < c->window_size) {
        c->history[c->index] = loss;
        c->sum += loss;
        c->count++;
    } else {
        c->sum -= c->history[c->index];
        c->history[c->index] = loss;
        c->sum += loss;
    }
    c->index = (c->index + 1) % c->window_size;


    if (loss + c->tolerance < c->min_loss) {
        c->min_loss = loss;
        c->no_improvement_count = 0;
    } else {
        c->no_improvement_count++;
    }
}

float training_controller_average_loss(const TrainingController_t c){
    if(c->count == 0){return 0.0f;}

    return c->sum / (float)c->count;
}

int training_controller_should_stop(const TrainingController_t c) {
    if (c->count < c->window_size) {
        return 0;
    }

    float avg_loss = c->sum / (float)c->window_size;
    if (avg_loss <= c->target_loss) {
        c->context = 1;
        return 1;
    }

    if (c->no_improvement_count >= c->patience) {
        c->context = 2;
        return 1;
    }

    return 0;
}

void train(Network_t NN, const float* inputs, const float* target) {
    forward_propagation(NN,inputs);
    backpropagation(NN,inputs,target);
}

int train_to_minimal_error(
    Network_t network,
    const float *inputs,
    const float *targets,
    int dataset_size,
    size_t input_size,
    size_t output_size,
    float learning_rate,
    int max_epoch,
    TrainingController_t controller
) {
    training_controller_reset(controller);
    set_learning_rate(network_get_optimizer(network),learning_rate);
    printf("Début de l'entrainement\n");

    for (size_t epoch = 0; epoch < max_epoch; epoch++) {
        float epoch_loss = 0.0f;

        for (int sample = 0; sample < dataset_size; sample++) {
            const float *input = inputs + sample * input_size;
            const float *target = targets + sample * output_size;
            train(network, input, target);
            epoch_loss += mse_loss(network, target);
        }
        epoch_loss /= dataset_size;
        training_controller_update(controller, epoch_loss);

        printf("Epoch %d terminé - Erreur : %f\n", epoch + 1, epoch_loss);

        if (training_controller_should_stop(controller)) {
            print_context(controller->context);
            return epoch + 1;
        }
    }
    printf("Max epoch achieved\n");
    return max_epoch;
}

int train_on_dataset_batch(
    Network_t network,
    dataset_t dataset,
    int output_size,
    float learning_rate,
    int max_epoch,
    TrainingController_t controller,
    bool shuffle
) {
    int input_size = dataset_get_inputSize(dataset);
    int dataset_size = dataset_get_size(dataset);
    int batch_size = dataset_get_batch_size(dataset);

    set_learning_rate(network_get_optimizer(network), learning_rate);

    float *batch_inputs = malloc(batch_size * input_size * sizeof(float));
    float *batch_targets = malloc(batch_size * output_size * sizeof(float));
    if (!batch_inputs || !batch_targets) {
        perror("Erreur d'allocation pour les buffers de batch");
        free(batch_inputs);
        free(batch_targets);
        return -1;
    }

    training_controller_reset(controller);

    for (int epoch = 0; epoch < max_epoch; epoch++) {
        float epoch_loss = 0.0f;
        int num_batches = (dataset_size + batch_size - 1) / batch_size;

        if (shuffle) {
            shuffle_batch(dataset);
        }

        for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
            int current_batch_size = (batch_idx == num_batches - 1) ?
                (dataset_size % batch_size) : batch_size;
            if (current_batch_size == 0) current_batch_size = batch_size;

            float *all_inputs = get_data_inputs(dataset);
            float *all_targets = get_data_labels_as_output(dataset, output_size, network_get_layer(network, 1));

            for (int i = 0; i < current_batch_size; i++) {
                int data_idx = batch_idx * batch_size + i;
                memcpy(batch_inputs + i * input_size, all_inputs + data_idx * input_size,
                       input_size * sizeof(float));
                memcpy(batch_targets + i * output_size, all_targets + data_idx * output_size,
                       output_size * sizeof(float));
            }

            free(all_inputs);
            free(all_targets);

            train_batch(network, batch_inputs, batch_targets,
                       current_batch_size, input_size, output_size);

            float batch_loss = 0.0f;
            for (int i = 0; i < current_batch_size; i++) {
                const float *input = batch_inputs + i * input_size;
                const float *target = batch_targets + i * output_size;
                forward_propagation(network, input);
                batch_loss += mse_loss(network, target);
            }
            batch_loss /= current_batch_size;
            epoch_loss += batch_loss * current_batch_size;  // Accumule Σ mse_i
        }

        epoch_loss /= dataset_size;  // Moyenne finale sur tout le dataset
        training_controller_update(controller, epoch_loss);

        printf("Epoch %d terminé - Erreur moyenne : %f\n", epoch + 1, epoch_loss);

        if (training_controller_should_stop(controller)) {
            print_context(controller->context);
            free(batch_inputs);
            free(batch_targets);
            return epoch + 1;
        }
    }

    free(batch_inputs);
    free(batch_targets);
    printf("Max epoch achieved\n");
    return max_epoch;
}

void train_batch(
    Network_t network,
    const float *batch_inputs,
    const float *batch_targets,
    int batch_size,
    int input_size,
    int output_size
) {
    // Réinitialiser UNIQUEMENT les deltas (gradients) pour le batch
    for (int l = 1; l < network_get_num_layers(network); l++) {
        Layer_t layer = network_get_layer(network, l);
        int size = layer_get_size(layer);
        memset(layer_get_deltas(layer), 0, size * sizeof(float));
    }

    // Accumuler les gradients sur tout le batch
    for (int i = 0; i < batch_size; i++) {
        const float *input = batch_inputs + i * input_size;
        const float *target = batch_targets + i * output_size;

        // Forward propagation
        forward_propagation(network, input);

        // Calculer les deltas pour la couche de sortie
        Layer_t out_layer = network_get_layer(network, network_get_num_layers(network) - 1);
        for (int j = 0; j < output_size; j++) {
            float* out_layer_values = layer_get_values(out_layer);
            float* out_layer_deltas = layer_get_deltas(out_layer);
            ActivationFunction out_activation = layer_get_activation(out_layer);
            float error = out_layer_values[j] - target[j];
            out_layer_deltas[j] += error * out_activation.derivative(out_layer_values[j]);
        }

        // Backpropagation (calcule les deltas pour toutes les couches)
        for (int l = network_get_num_layers(network) - 2; l >= 0; l--) {
            Layer_t cur = network_get_layer(network, l);
            Layer_t next = network_get_layer(network, l + 1);
            float* next_weights = layer_get_weights(next);
            float* next_deltas = layer_get_deltas(next);
            float* cur_values = layer_get_values(cur);
            float* cur_deltas = layer_get_deltas(cur);
            ActivationFunction cur_activation = layer_get_activation(cur);
            int input_size_cur = layer_get_size(cur);
            int output_size_next = layer_get_size(next);

            for (int in = 0; in < input_size_cur; in++) {
                float sum = 0.0f;
                for (int out = 0; out < output_size_next; out++) {
                    sum += next_weights[out * input_size_cur + in] * next_deltas[out];
                }
                cur_deltas[in] += sum * cur_activation.derivative(cur_values[in]);
            }
        }
    }

    // Normaliser les deltas par la taille du batch (CORRECTION BUG #2)
    for (int l = 1; l < network_get_num_layers(network); l++) {
        Layer_t layer = network_get_layer(network, l);
        int size = layer_get_size(layer);
        for (int i = 0; i < size; i++) {
            layer_get_deltas(layer)[i] /= (float)batch_size;
        }
    }

    // Mise à jour des poids (une seule fois pour tout le batch)
    for (int l = 1; l < network_get_num_layers(network); l++) {
        Layer_t cur = network_get_layer(network, l);
        Layer_t prev = network_get_layer(network, l - 1);
        float* prev_values = layer_get_values(prev);
        network_get_optimizer(network)->update_layer(network_get_optimizer(network), cur, prev_values);
    }
}

#define WASHOUT_SIZE  2000
#define TRAIN_SIZE    8000
#define TEST_SIZE     2000

#define TRAIN_START   WASHOUT_SIZE
#define TEST_START    (TRAIN_START + TRAIN_SIZE)
void train_mackey_glass(
    Network_t NN,
    const float *series,
    size_t sample_count
) {
    if (!NN || !series || sample_count <= TEST_START + 1) {
        return;
    }

    // 1. Washout
    for (size_t i = 0; i < WASHOUT_SIZE; ++i) {
        forward_propagation(NN, &series[i]);
    }

    // 2. Training
    for (size_t i = TRAIN_START; i < TEST_START; ++i) {
        float input = series[i];
        float target = series[i + 1];

        train(NN, &input, &target);
    }
}