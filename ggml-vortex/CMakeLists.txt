find_package(OpenCL)

set(TARGET_NAME ggml-vortex)

ggml_add_backend_library(${TARGET_NAME}
    ggml-vortex.cpp
    ../../include/ggml-vortex.h
)

if(OpenCL_FOUND)
    target_link_libraries(${TARGET_NAME} PRIVATE ${OpenCL_LIBRARIES})
    target_include_directories(${TARGET_NAME} PRIVATE ${OpenCL_INCLUDE_DIRS})
    target_compile_definitions(${TARGET_NAME} PRIVATE GGML_VORTEX_USE_OPENCL)
    message(STATUS "Vortex backend: OpenCL found, will use for kernels")
else()
    message(STATUS "Vortex backend: OpenCL not found, using CPU fallback")
endif()

target_compile_definitions(${TARGET_NAME} PRIVATE GGML_USE_VORTEX)