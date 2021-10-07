# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_executable(MNIST_cxx "MNIST.cc")
target_include_directories(MNIST_cxx PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS})
target_link_libraries(MNIST_cxx PRIVATE onnxruntime ${PNG_LIBRARIES})
if(PNG_LIBDIR)
	target_link_directories(MNIST_cxx PRIVATE ${PNG_LIBDIR})
endif()
