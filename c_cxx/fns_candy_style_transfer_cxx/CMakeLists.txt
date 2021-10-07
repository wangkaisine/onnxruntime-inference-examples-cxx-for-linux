# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

add_executable(fns_candy_style_transfer_cxx "fns_candy_style_transfer.cc")
target_include_directories(fns_candy_style_transfer_cxx PRIVATE ${PROJECT_SOURCE_DIR}/include ${PNG_INCLUDE_DIRS})
target_link_libraries(fns_candy_style_transfer_cxx PRIVATE onnxruntime ${PNG_LIBRARIES})
if(PNG_LIBDIR)
  target_link_directories(fns_candy_style_transfer_cxx PRIVATE ${PNG_LIBDIR})
endif()
