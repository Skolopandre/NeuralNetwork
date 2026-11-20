
cmake_minimum_required(VERSION 3.29)

project(NeuralNetwork C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

add_executable(NeuralNetwork
        src/core/main.c
        src/core/network.c
        src/core/network_builder.c
        src/core/training.c
        src/optimizer/optimizer.c
        src/optimizer/adamw.c
        src/optimizer/sgd.c
        src/activationFunction/lrelu.c
        src/activationFunction/relu.c
        src/activationFunction/tanh.c
        src/activationFunction/linear.c
        src/Dataloader/DataLoader.c
        src/layer/convolution.c
        src/layer/layer.c
        src/layer/reservoir.c
        src/layer/dense.c
        src/layer/flatten.c
        src/activationFunction/softplus.c

)

# Répertoires des headers
target_include_directories(NeuralNetwork PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(NeuralNetwork PROPERTIES
        VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
        RUNTIME_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/bin"
)

file(COPY ${PROJECT_SOURCE_DIR}/data
        DESTINATION ${PROJECT_BINARY_DIR}/bin
)

# Optimisations
target_compile_options(NeuralNetwork PRIVATE
        -O3
        -march=native
)
