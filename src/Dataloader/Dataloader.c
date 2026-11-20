
#include "DataLoader.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <string.h>
#include <limits.h>
#include <time.h>

#define MNIST_IMAGE_MAGIC 2051u
#define MNIST_LABEL_MAGIC 2049u

struct dataset {
    uint32_t size;
    uint32_t batch_size;
    uint32_t input_size;

    uint8_t *data;
    uint32_t *labels;
};

/* --------------------------------------------------------- */
/* Utilitaires                                               */
/* --------------------------------------------------------- */

static bool checked_multiply_size(size_t a, size_t b, size_t *result) {
    if (!result || (b != 0 && a > SIZE_MAX / b)) {
        return false;
    }

    *result = a * b;
    return true;
}

static bool read_uint32_be(FILE *file, uint32_t *value) {
    uint8_t bytes[4];

    if (!file || !value || fread(bytes, 1, sizeof(bytes), file) != sizeof(bytes)) {
        return false;
    }

    *value = ((uint32_t)bytes[0] << 24)
           | ((uint32_t)bytes[1] << 16)
           | ((uint32_t)bytes[2] << 8)
           |  (uint32_t)bytes[3];

    return true;
}

static bool read_image_header(
    FILE *file,
    uint32_t *count,
    uint32_t *rows,
    uint32_t *cols
) {
    uint32_t magic;

    if (!read_uint32_be(file, &magic)
        || !read_uint32_be(file, count)
        || !read_uint32_be(file, rows)
        || !read_uint32_be(file, cols)) {
        return false;
    }

    if (magic != MNIST_IMAGE_MAGIC || *count == 0 || *rows == 0 || *cols == 0) {
        return false;
    }

    return true;
}

static bool read_label_header(FILE *file, uint32_t *count) {
    uint32_t magic;

    if (!read_uint32_be(file, &magic)
        || !read_uint32_be(file, count)) {
        return false;
    }

    return magic == MNIST_LABEL_MAGIC && *count > 0;
}

static dataset_t dataset_allocate(
    uint32_t size,
    uint32_t batch_size,
    uint32_t input_size
) {
    if (size == 0 || input_size == 0) {
        return NULL;
    }

    size_t data_count;
    size_t data_bytes;
    size_t label_bytes;

    if (!checked_multiply_size(size, input_size, &data_count)
        || !checked_multiply_size(data_count, sizeof(uint8_t), &data_bytes)
        || !checked_multiply_size(size, sizeof(uint32_t), &label_bytes)) {
        return NULL;
    }

    dataset_t dataset = calloc(1, sizeof(*dataset));

    if (!dataset) {
        return NULL;
    }

    dataset->data = malloc(data_bytes);
    dataset->labels = malloc(label_bytes);

    if (!dataset->data || !dataset->labels) {
        free(dataset->data);
        free(dataset->labels);
        free(dataset);
        return NULL;
    }

    dataset->size = size;
    dataset->batch_size = batch_size;
    dataset->input_size = input_size;

    return dataset;
}

/* --------------------------------------------------------- */
/* Initialisation depuis les fichiers IDX                    */
/* --------------------------------------------------------- */

dataset_t dataset_init_from_file(
    char *data_file,
    char *label_file,
    uint32_t batch_size,
    bool randomize
) {
    if (!data_file || !label_file) {
        return NULL;
    }

    FILE *image_fp = fopen(data_file, "rb");

    if (!image_fp) {
        perror("Impossible d'ouvrir le fichier images");
        return NULL;
    }

    FILE *label_fp = fopen(label_file, "rb");

    if (!label_fp) {
        perror("Impossible d'ouvrir le fichier labels");
        fclose(image_fp);
        return NULL;
    }

    uint32_t image_count, rows, cols;
    uint32_t label_count;

    if (!read_image_header(image_fp, &image_count, &rows, &cols)) {
        fprintf(stderr, "Header IDX images invalide\n");
        fclose(image_fp);
        fclose(label_fp);
        return NULL;
    }

    if (!read_label_header(label_fp, &label_count)) {
        fprintf(stderr, "Header IDX labels invalide\n");
        fclose(image_fp);
        fclose(label_fp);
        return NULL;
    }

    if (image_count != label_count) {
        fprintf(stderr,
                "Nombre d'images (%u) différent du nombre de labels (%u)\n",
                image_count, label_count);
        fclose(image_fp);
        fclose(label_fp);
        return NULL;
    }

    size_t input_size;

    if (!checked_multiply_size(rows, cols, &input_size)
        || input_size > UINT32_MAX) {
        fprintf(stderr, "Dimensions des images invalides\n");
        fclose(image_fp);
        fclose(label_fp);
        return NULL;
    }

    dataset_t dataset = dataset_allocate(
        image_count,
        batch_size,
        (uint32_t)input_size
    );

    if (!dataset) {
        fprintf(stderr, "Allocation du dataset impossible\n");
        fclose(image_fp);
        fclose(label_fp);
        return NULL;
    }

    size_t image_bytes;
    size_t label_bytes;

    if (!checked_multiply_size(image_count, input_size, &image_bytes)
        || !checked_multiply_size(label_count, sizeof(uint8_t), &label_bytes)) {
        dataset_free(dataset);
        fclose(image_fp);
        fclose(label_fp);
        return NULL;
    }

    if (fread(dataset->data, 1, image_bytes, image_fp) != image_bytes) {
        fprintf(stderr, "Erreur lors de la lecture des images\n");
        dataset_free(dataset);
        fclose(image_fp);
        fclose(label_fp);
        return NULL;
    }

    /*
     * Les labels IDX sont stockés sur un octet.
     * On les convertit vers uint32_t dans le dataset.
     */
    uint8_t *raw_labels = malloc(label_bytes);

    if (!raw_labels) {
        dataset_free(dataset);
        fclose(image_fp);
        fclose(label_fp);
        return NULL;
    }

    if (fread(raw_labels, 1, label_bytes, label_fp) != label_bytes) {
        fprintf(stderr, "Erreur lors de la lecture des labels\n");
        free(raw_labels);
        dataset_free(dataset);
        fclose(image_fp);
        fclose(label_fp);
        return NULL;
    }

    for (uint32_t i = 0; i < label_count; ++i) {
        dataset->labels[i] = raw_labels[i];
    }

    free(raw_labels);
    fclose(image_fp);
    fclose(label_fp);

    if (randomize) {
        shuffle_batch(dataset);
    }

    return dataset;
}

/* --------------------------------------------------------- */
/* Initialisation depuis la mémoire                          */
/* --------------------------------------------------------- */

dataset_t dataset_init_from_memory(
    uint8_t *data,
    uint32_t *labels,
    uint32_t dataset_size,
    uint32_t batch_size,
    uint32_t input_size,
    bool randomize
) {
    if (!data || !labels || dataset_size == 0 || input_size == 0) {
        return NULL;
    }

    dataset_t dataset = dataset_allocate(
        dataset_size,
        batch_size,
        input_size
    );

    if (!dataset) {
        return NULL;
    }

    size_t data_count;
    size_t data_bytes;

    if (!checked_multiply_size(dataset_size, input_size, &data_count)
        || !checked_multiply_size(data_count, sizeof(uint8_t), &data_bytes)) {
        dataset_free(dataset);
        return NULL;
    }

    memcpy(dataset->data, data, data_bytes);
    memcpy(dataset->labels, labels, (size_t)dataset_size * sizeof(*labels));

    if (randomize) {
        shuffle_batch(dataset);
    }

    return dataset;
}

/* --------------------------------------------------------- */
/* Libération                                                */
/* --------------------------------------------------------- */

void dataset_free(dataset_t dataset) {
    if (!dataset) {
        return;
    }

    free(dataset->data);
    free(dataset->labels);
    free(dataset);
}

/* --------------------------------------------------------- */
/* Mélange des données                                       */
/* --------------------------------------------------------- */

void shuffle_batch(dataset_t dataset) {
    if (!dataset || dataset->size <= 1) {
        return;
    }

    for (size_t i = dataset->size - 1; i > 0; --i) {
        size_t j = (size_t)rand() % (i + 1);

        if (i == j) {
            continue;
        }

        size_t offset_i = i * dataset->input_size;
        size_t offset_j = j * dataset->input_size;

        for (size_t k = 0; k < dataset->input_size; ++k) {
            uint8_t tmp = dataset->data[offset_i + k];
            dataset->data[offset_i + k] = dataset->data[offset_j + k];
            dataset->data[offset_j + k] = tmp;
        }

        uint32_t tmp_label = dataset->labels[i];
        dataset->labels[i] = dataset->labels[j];
        dataset->labels[j] = tmp_label;
    }
}

/* --------------------------------------------------------- */
/* Extraction des entrées                                    */
/* --------------------------------------------------------- */

float *get_data_inputs(dataset_t dataset) {
    if (!dataset) {
        return NULL;
    }

    size_t count;
    size_t bytes;

    if (!checked_multiply_size(dataset->size, dataset->input_size, &count)
        || !checked_multiply_size(count, sizeof(float), &bytes)) {
        return NULL;
    }

    float *batch_data = malloc(bytes);

    if (!batch_data) {
        return NULL;
    }

    for (size_t i = 0; i < count; ++i) {
        batch_data[i] = (float)dataset->data[i] / 255.0f;
    }

    return batch_data;
}

/* --------------------------------------------------------- */
/* Extraction des labels sous forme de vecteurs              */
/* --------------------------------------------------------- */

float *get_data_labels_as_output(
    dataset_t dataset,
    uint32_t output_size,
    Layer_t output_layer
) {
    if (!dataset || !output_layer || output_size == 0) {
        return NULL;
    }

    size_t count;
    size_t bytes;

    if (!checked_multiply_size(dataset->size, output_size, &count)
        || !checked_multiply_size(count, sizeof(float), &bytes)) {
        return NULL;
    }

    ActivationFunction activation = layer_get_activation(output_layer);

    float label_limit = (activation.activation == tanh_activation)
                      ? -1.0f
                      : 0.0f;

    float *batch_labels = malloc(bytes);

    if (!batch_labels) {
        return NULL;
    }

    for (size_t i = 0; i < dataset->size; ++i) {
        if (dataset->labels[i] >= output_size) {
            fprintf(stderr, "Label %u hors limites pour output_size=%u\n",
                    dataset->labels[i], output_size);
            free(batch_labels);
            return NULL;
        }

        for (size_t j = 0; j < output_size; ++j) {
            batch_labels[i * output_size + j] =
                (dataset->labels[i] == j) ? 1.0f : label_limit;
        }
    }

    return batch_labels;
}

/* --------------------------------------------------------- */
/* Extraction des labels sous forme scalaire                 */
/* --------------------------------------------------------- */

float *get_data_labels(dataset_t dataset) {
    if (!dataset) {
        return NULL;
    }

    size_t bytes;

    if (!checked_multiply_size(dataset->size, sizeof(float), &bytes)) {
        return NULL;
    }

    float *batch_labels = malloc(bytes);

    if (!batch_labels) {
        return NULL;
    }

    for (size_t i = 0; i < dataset->size; ++i) {
        batch_labels[i] = (float)dataset->labels[i];
    }

    return batch_labels;
}

/* --------------------------------------------------------- */
/* Getters                                                   */
/* --------------------------------------------------------- */

uint32_t dataset_get_size(dataset_t dataset) {
    return dataset ? dataset->size : 0;
}

uint32_t dataset_get_batch_size(dataset_t dataset) {
    return dataset ? dataset->batch_size : 0;
}

uint32_t dataset_get_inputSize(dataset_t dataset) {
    return dataset ? dataset->input_size : 0;
}

/* --------------------------------------------------------- */
/* Affichage MNIST                                           */
/* --------------------------------------------------------- */

void print_mnist_image(
    const float *image,
    int width,
    int height,
    int label,
    size_t prediction
) {
    if (!image || width <= 0 || height <= 0) {
        return;
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float pixel = image[(size_t)y * (size_t)width + (size_t)x];

            if (pixel > 0.8f) {
                printf("##");
            } else if (pixel > 0.4f) {
                printf("++");
            } else if (pixel > 0.1f) {
                printf("..");
            } else {
                printf("  ");
            }
        }

        printf("\n");
    }

    printf("Label      : %d\n", label);
    printf("Prediction : %zu\n", prediction);
}
