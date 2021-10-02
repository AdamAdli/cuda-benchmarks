
cmake_minimum_required (VERSION 3.10)
project (Test CXX CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(abseil-cpp)
find_package(CUDA)

set_source_files_properties(test.cu.cc PROPERTIES LANGUAGE CUDA)


add_executable(test test.cu.cc)
target_include_directories(test PUBLIC . /opt/cuda/targets/x86_64-linux/include/)
# target_include_directories(test PUBLIC .)
target_link_libraries(test cudart_static cusparse_static culibos absl::base absl::random_random)

