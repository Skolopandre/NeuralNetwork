#include "optimizer.h"

Optimizer_t create_optimizer_from_type(OptimizerType type)
{
    switch (type)
    {
        case SGD:
            return optimizer_create_sgd(0.1f);

        case ADAMW:
            return optimizer_create_adamw(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);

        default:
            fprintf(stderr, "Optimiseur ADAMW par défaut\n");
        return optimizer_create_adamw(0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);
    }
}

void set_learning_rate(Optimizer_t opt,float learning_rate){
    if (!opt || !opt->context)
        return;

    switch (opt->type) {
        case SGD:
            ((SGDContext *)opt->context)->learning_rate =
                learning_rate;
        break;

        case ADAMW:
            ((AdamWContext *)opt->context)->learning_rate =
                learning_rate;
        break;
    }
}

void free_optimizer(Optimizer *opt)
{
    if (opt)
    {
        free(opt->context);
        free(opt);
    }
}