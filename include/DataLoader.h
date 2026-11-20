#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdint.h>
#include <stdbool.h>
#include "network.h"


typedef struct dataset dataset;
typedef struct dataset *dataset_t;

dataset_t dataset_init_from_file(
    char *data_file,
    char *label_file,
    uint32_t batch_size,
    bool randomize
);

dataset_t dataset_init_from_memory(
    uint8_t *data,
    uint32_t *labels,
    uint32_t dataset_size,
    uint32_t batch_size,
    uint32_t input_size,
    bool randomize
);

void dataset_free(dataset_t dataset);
void shuffle_batch(dataset_t dataset);

float *get_data_inputs(dataset_t dataset);
float *get_data_labels(dataset_t dataset);

float *get_data_labels_as_output(
    dataset_t dataset,
    uint32_t output_size,
    Layer_t output_layer
);

uint32_t dataset_get_size(dataset_t dataset);
uint32_t dataset_get_batch_size(dataset_t dataset);
uint32_t dataset_get_inputSize(dataset_t dataset);

void print_mnist_image(
    const float *image,
    int width,
    int height,
    int label,
    size_t prediction
);
#endif //DATALOADER_H