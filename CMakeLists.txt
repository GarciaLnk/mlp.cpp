cmake_minimum_required(VERSION 3.5.0)
project(mlp VERSION 0.1.0 LANGUAGES CXX) 

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -flto=auto -Wno-unused")

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
    add_compile_options(-march=x86-64-v3)
endif()

include(CTest)
enable_testing()

# Source files for the MLP library
set(MLP_SOURCES
    src/mlp.cpp
    src/layer.cpp
    src/neuron.cpp
    src/utils.cpp
)

add_library(mlp STATIC ${MLP_SOURCES})
target_include_directories(mlp PUBLIC include)

# Executable for the Iris example
add_executable(iris_train examples/iris_train.cpp)
add_executable(iris_predict examples/iris_predict.cpp)
target_include_directories(iris_train PRIVATE include)
target_include_directories(iris_predict PRIVATE include)
target_link_libraries(iris_train mlp)
target_link_libraries(iris_predict mlp)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples/iris.csv DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/examples/iris_model.bin DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Tests
add_executable(neuron_test tests/neuron_test.cpp)
target_include_directories(neuron_test PRIVATE include)
target_link_libraries(neuron_test mlp)

add_executable(layer_test tests/layer_test.cpp)
target_include_directories(layer_test PRIVATE include)
target_link_libraries(layer_test mlp)

add_executable(mlp_test tests/mlp_test.cpp)
target_include_directories(mlp_test PRIVATE include)
target_link_libraries(mlp_test mlp)

add_test(NAME NeuronTest COMMAND neuron_test)
add_test(NAME LayerTest COMMAND layer_test)
add_test(NAME MLPTest COMMAND mlp_test)