#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <stdio.h>
#include <stdlib.h>

typedef struct Layer Layer;
typedef Layer *Layer_t;

typedef enum{
    SGD,
    ADAMW
} OptimizerType;

/* Contexte générique d’un optimiseur */
typedef struct Optimizer
{
    OptimizerType type;

    void (*update_layer)(
        struct Optimizer *opt,
        Layer *layer,
        const float *prev_values);

    void (*set_learning_rate)(
        struct Optimizer *opt,
        float learning_rate
    );

    void *context; // données spécifiques à l’optimiseur
} Optimizer;
typedef Optimizer *Optimizer_t;

typedef struct AdamWContext{
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    int step;
}AdamWContext;

typedef struct SGDContext{
    float learning_rate;
}SGDContext;



Optimizer_t optimizer_create_sgd(float learning_rate);
Optimizer_t optimizer_create_adamw(float learning_rate,float beta1, float beta2, float epsilon, float weight_decay);
Optimizer_t create_optimizer_from_type(OptimizerType type);

void free_optimizer(Optimizer *opt);

void set_learning_rate(Optimizer_t opt, float learning_rate);

void step_Adamw(Optimizer *opt);

#endif