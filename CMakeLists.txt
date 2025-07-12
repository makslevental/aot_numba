project(aot_numba)

set(CMAKE_CXX_STANDARD 20)

add_executable(aot_numba main.cpp)

find_package (Python3 COMPONENTS Interpreter Development)

add_library(my_module SHARED IMPORTED)
set_target_properties(my_module PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/my_module.so)

target_link_libraries(aot_numba PRIVATE my_module ${Python3_LIBRARIES})
