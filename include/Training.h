#ifndef TRAINING_H
#define TRAINING_H

#include "network.h"
#include "DataLoader.h"

typedef struct TrainingController TrainingController;
typedef TrainingController* TrainingController_t;

/* Construction */

TrainingController_t create_training_controller(
size_t window_size,
float target_loss,
size_t patience,          // Nombre d'itérations sans amélioration
float tolerance        // Tolérance pour considérer qu'il n'y a pas d'amélioration
);

void free_training_controller(
    TrainingController_t controller
);

/* Réinitialisation */

void training_controller_reset(
    TrainingController_t controller
);

/* Ajout d'une nouvelle loss */

void training_controller_update(
    TrainingController_t controller,
    float loss
);

/* Etat */

float training_controller_average_loss(
    const TrainingController_t controller
);

int training_controller_should_stop(
    const TrainingController_t controller
);

/* Entraînement */

void train(Network_t NN, const float* inputs, const float* target);

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
);

int train_on_dataset_batch(
    Network_t network,
    dataset_t dataset,
    int output_size,
    float learning_rate,
    int max_epoch,
    TrainingController_t controller,
    bool shuffle
);

void train_batch(
    Network_t network,
    const float *batch_inputs,   // Pointeur vers les entrées du batch [batch_size * input_size]
    const float *batch_targets,  // Pointeurs vers les cibles du batch [batch_size * output_size]
    int batch_size,
    int input_size,
    int output_size
);

void train_mackey_glass(
    Network_t NN,
    const float *series,
    size_t sample_count
);


#endif