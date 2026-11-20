#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <windows.h>

#include "network.h"
#include "Training.h"
#include "DataLoader.h"
#include "network_builder.h"
#include "math.h"


#define FILE_WEIGHT "C:/Users/limer/CLionProjects/NeuralNetwork/data/weights.txt"
#define SAMPLE_COUNT 100

#define SAMPLE_COUNT 12000

#define WASHOUT_SIZE 2000
#define TRAIN_SIZE   8000
#define TEST_SIZE    2000

#define TRAIN_START  WASHOUT_SIZE
#define TEST_START   (TRAIN_START + TRAIN_SIZE)

#define RESERVOIR_SIZE 500

static void run_mackey_glass_test(Network_t NN, const float *series)
{
    if (!NN || !series) {
        return;
    }

    printf("Starting Mackey-Glass experiment...\n");

    /*
     * 1. Reset reservoir state.
     */
    network_reset_reservoir(NN);

    /*
     * 2. Washout.
     */
    printf("Washout...\n");

    for (size_t i = 0; i < WASHOUT_SIZE; ++i) {
        forward_propagation(NN, &series[i]);
    }

    /*
     * 3. Training.
     */
    printf("Training...\n");

    for (size_t i = TRAIN_START; i < TEST_START; ++i) {

        float input = series[i];
        float target = series[i + 1];

        train(NN, &input, &target);

        if ((i - TRAIN_START + 1) % 1000 == 0) {
            printf("Training: %zu / %d\n",
                   i - TRAIN_START + 1,
                   TRAIN_SIZE);
        }
    }

    /*
     * 4. Testing.
     */
    printf("Testing...\n");

    float squared_error = 0.0f;

    for (size_t i = TEST_START; i < SAMPLE_COUNT - 1; ++i) {

        float input = series[i];
        float target = series[i + 1];

        forward_propagation(NN, &input);

        const float *output =
            layer_get_values(network_get_layer(NN,network_get_num_layers(NN)-1));

        if (!output) {
            fprintf(stderr, "Invalid network output.\n");
            return;
        }

        float prediction = output[0];
        float error = prediction - target;

        squared_error += error * error;

        if ((i - TEST_START) % 200 == 0) {
            printf("Target: %.6f | Prediction: %.6f\n",
                   target,
                   prediction);
        }
    }

    const float rmse = sqrtf(squared_error / TEST_SIZE);

    printf("\nExperiment completed.\n");
    printf("RMSE: %.8f\n", rmse);
}

static bool mackey_glass_generate(
    float *output,
    size_t count,
    double dt,
    double beta,
    double gamma,
    double n,
    double tau,
    double initial_value
) {
    if (!output || count == 0 ||
        !isfinite(dt) || dt <= 0.0 ||
        !isfinite(beta) || beta <= 0.0 ||
        !isfinite(gamma) || gamma <= 0.0 ||
        !isfinite(n) || n <= 0.0 ||
        !isfinite(tau) || tau <= 0.0 ||
        !isfinite(initial_value)) {
        return false;
        }

    // The delay must correspond to an integer number of samples.
    double delay_samples = tau / dt;
    double rounded_delay = round(delay_samples);

    if (fabs(delay_samples - rounded_delay) > 1e-9 ||
        rounded_delay < 1.0 ||
        rounded_delay > (double)SIZE_MAX) {
        return false;
        }

    size_t delay = (size_t)rounded_delay;

    // Initial condition: x(t) = initial_value for t <= 0.
    output[0] = (float)initial_value;

    for (size_t i = 0; i + 1 < count; ++i) {

        // Retrieve the delayed value.
        // Before the delay has elapsed, use the initial history.
        double delayed_value = initial_value;

        if (i >= delay) {
            delayed_value = output[i - delay];
        }

        double current_value = output[i];

        // Mackey-Glass differential equation.
        double derivative =
            beta * delayed_value /
            (1.0 + pow(delayed_value, n))
            - gamma * current_value;

        // Explicit Euler integration.
        double next_value = current_value + dt * derivative;

        if (!isfinite(next_value) ||
            !isfinite((float)next_value)) {
            return false;
            }

        output[i + 1] = (float)next_value;
    }

    return true;
}

void benchmark() {
    network_builder_t builder = builder_init(128);
    builder_add_dense(builder,256,TANH);
    builder_add_dense(builder,256,TANH);
    builder_add_dense(builder,10,TANH);

    Network_t NN = init_network(builder,ADAMW);
    builder_destroy(builder);

    float inputs[1000][128];
    float targets[1000][10];

    for(int n = 0; n < 1000; n++){

        for(int i = 0; i < 128; i++){
            inputs[n][i] = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
        }


        int class = 0;
        float best = -999999.0f;

        for(int c = 0; c < 10; c++){
            float sum = 0;
            int start = c * (128 / 10);

            for(int j = 0; j < 128 / 10; j++){
                sum += inputs[n][start+j];
            }

            if(sum > best){
                best = sum;
                class = c;
            }
        }

        for(int c = 0; c < 10; c++){
            targets[n][c] = 0.0f;
        }

        targets[n][class] = 1.0f;
    }



    printf("Debut entrainement...\n");
    clock_t begin = clock();

    for(int epoch = 0; epoch < 1000; epoch++){
        for(int n = 0; n < 1000; n++){
            train(
                NN,
                inputs[n],
                targets[n]
            );
        }


        if(epoch % 100 == 0){
            printf("Epoch %d\n", epoch);
        }
    }


    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

    printf("\nTemps total : %f s\n", time_spent);
    printf("Temps / epoch : %f ms\n",time_spent / 1000 * 1000);
    printf("\nQuelques predictions :\n");
    for(int i = 0; i < 5; i++){

        int true_class = 0;
        float best = -999999.0f;
        for(int c = 0; c < 10; c++){
            float sum = 0;
            int start = c * (128 / 10);
            for(int j = 0; j < 128 / 10; j++){
                sum += inputs[i][start+j];
            }
            if(sum > best){
                best = sum;
                true_class = c;
            }
        }

        int pred = predict(NN);
        printf("Reel : %d | Predit : %d\n", true_class, pred);
    }

    save_weights(NN,FILE_WEIGHT);
    free_network(NN);

    time_t now = time(NULL);
    struct tm *t = localtime(&now);

    FILE *f = fopen("C:/Users/limer/CLionProjects/NeuralNetwork/data/Speed_benchmark.txt","a+");
    fprintf(f,"Date: %d ", t->tm_year + 1900);
    fprintf(f,"%d ", t->tm_mon + 1);
    fprintf(f,"%d\n", t->tm_mday);
    fprintf(f,"Time: %d:", t->tm_hour);
    fprintf(f,"%d\n", t->tm_min);
    fprintf(f,"time spent : %f\n-----------------\n",time_spent);

    fclose(f);
}

void XOR() {
    network_builder_t builder = builder_init(2);
    builder_add_dense(builder,4,LRELU);
    builder_add_dense(builder,1,LRELU);
    Network_t NN = init_network(builder,ADAMW);
    builder_destroy(builder);

    int choix;
    printf("Souhaitez-vous load les weights existants depuis %s\n(1)oui - (2)non\n",FILE_WEIGHT);
    scanf("%d",&choix);
    if(choix==1){
        load_weights(NN,FILE_WEIGHT);
    }

    float inputs[4][2] = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    float targets[4][1] = {
        {0},
        {1},
        {1},
        {0}
    };

    printf("Avant entrainement :\n");
    for (int i = 0; i < 4; i++) {
        int pred = predict(NN);
        printf("Entree : %.0f %.0f => Sortie predite : %d\n", inputs[i][0], inputs[i][1], pred);
    }

    clock_t begin = clock();
    TrainingController_t controller = create_training_controller(
        10,        // moyenne mobile sur 10 epochs
        0.0005f,   // target_loss
        1000,        // nombre d'epochs sans amélioration avant arrêt
        0.005f     // variation minimale pour considérer une amélioration
    );

    int epochs = train_to_minimal_error(
        NN,
        &inputs[0][0],
        &targets[0][0],
        4,      // dataset_size
        2,      // input_size
        1,      // output_size
        0.1f, // learning_rate
        3000, // max_epoch
        controller
    );

    printf("Convergence apres %d epochs\n", epochs);

    free_training_controller(controller);
    clock_t end = clock();
    double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;


    printf("\nAprès entrainement :\n");
    for (int i = 0; i < 4; i++) {
        float output[1];
        predict_full(NN, inputs[i], output);
        printf("Entree : %.0f %.0f => Sortie brute : %.4f, predite : %d\n",
               inputs[i][0], inputs[i][1], output[0], output[0] > 0.5f ? 1 : 0);
    }

    save_weights(NN,FILE_WEIGHT);
    free_network(NN);
    printf("temps total : %fs\nTemps par epoch : %.3f microsecondes",time_spent,time_spent/epochs*1000*1000);
}


void MNIST(char* data_file, char* label_file,char* test_data_file, char* test_labels_file){
    network_builder_t builder = builder_init(28*28);
    builder_add_dense(builder,128,LRELU);
    builder_add_dense(builder,10,LRELU);


    Network_t NN = init_network(builder,ADAMW);

    dataset_t training_data = dataset_init_from_file(data_file, label_file, 60000, true);

    TrainingController_t controller = create_training_controller(
        3,        // moyenne mobile sur n epochs
        0.004f,   // target_loss
        3,        // nombre d'epochs sans amélioration avant arrêt
        0.001f     // variation minimale pour considérer une amélioration
    );

    float *inputs = get_data_inputs(training_data);
    float *targets = get_data_labels_as_output(training_data,10,network_get_layer(NN,1));

    time_t train_begin = clock();
    int epochs = train_to_minimal_error(
        NN,
        inputs,
        targets,
        60000,
        28*28,
        10,
        0.0003,
        20,
        controller);
    time_t train_end = clock();

    printf("Epoch pour convergence : %d\n",epochs);
    printf("Temps total : %.3fs\nTemps par epoch : %.3fs\n",(double)(train_end-train_begin)/CLOCKS_PER_SEC,(double)(train_end-train_begin)/CLOCKS_PER_SEC/epochs);
    free_training_controller(controller);
    free(inputs);
    free(targets);
    dataset_free(training_data);

    save_weights(NN,FILE_WEIGHT);

    dataset_t test_data = dataset_init_from_file(test_data_file,test_labels_file,10000,true);
    float* test_inputs = get_data_inputs(test_data);
    float* test_targets = get_data_labels(test_data);
    size_t batch_size = dataset_get_size(test_data);
    size_t stride = dataset_get_inputSize(test_data);
    int got_it = 0;


    for(size_t i=0;i<batch_size;i++){
        float outputs[10];
        predict_full(NN, test_inputs + i * stride, outputs);
        size_t prediction = 0;
        for (size_t j = 1; j < 10; ++j) {
            if (outputs[j] > outputs[prediction]) {
                prediction = j;
            }
        }

        //printf("Label: %u | Prediction: %zu\n",(uint32_t)test_targets[i],prediction);
        if((uint32_t)test_targets[i] == prediction) {
            got_it++;
        }
        /*else {
            for(int j=0;j<10;j++) {
                printf("[%d] -->  [%f]\n",j,outputs[j]);
            }
            //print_mnist_image(test_inputs + i * stride,28,28,(int)test_targets[i],prediction);
        }
        */
    }
    printf("Accuracy : %.2f%%\n", 100.0f * (float)got_it / batch_size);


    free(test_inputs);
    free(test_targets);
    dataset_free(test_data);
    free_network(NN);
}

void reservoirTest() {
    float *series = malloc(SAMPLE_COUNT * sizeof(float));

    if (!series) {
        return ;
    }

    if (!mackey_glass_generate(
            series,
            SAMPLE_COUNT,
            0.1,
            0.2,
            0.1,
            10.0,
            17.0,
            1.2)) {

        free(series);
        return ;
            }

    network_builder_t builder = builder_init(1);

    builder_add_reservoir(
        builder,
        RESERVOIR_SIZE,
        TANH,
        0.6f,
        0.2f,
        0.9f
    );

    builder_add_dense(builder, 1, LRELU);

    Network_t NN = init_network(builder, ADAMW);

    if (!NN) {
        free(series);
        return ;
    }

    run_mackey_glass_test(NN, series);

    free_network(NN);
    free(series);

}

int main(){
    SetConsoleOutputCP(CP_UTF8);
#ifdef __AVX2__
    printf("AVX2 enabled\n");
#else
    printf("Scalar version\n");
#endif
    srand(time(NULL));

    int choix;


    printf("Que souhaitez-vous faire? \n"
           "(1) - benchmark speed\n"
           "(2) - Tester XOR\n"
           "(3) - MNIST\n"
           "(4) - test reservoir\n");
    scanf("%d",&choix);
    switch(choix) {
        case 1:
            benchmark();
            break;
        case 2:
            XOR();
            break;
        case 3:
            MNIST("C:/Users/limer/CLionProjects/NeuralNetwork/data/train-images.idx3-ubyte",
                "C:/Users/limer/CLionProjects/NeuralNetwork/data/train-labels.idx1-ubyte",
                "C:/Users/limer/CLionProjects/NeuralNetwork/data/t10k-images.idx3-ubyte",
                "C:/Users/limer/CLionProjects/NeuralNetwork/data/t10k-labels.idx1-ubyte");
            break;
        case 4 :
            reservoirTest();
            break;
    }

    return 0;
}
