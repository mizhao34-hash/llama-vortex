set(TARGET_NAME ggml-vortex)

set(VORTEX_RUNTIME_DIR /home/cs254A/vortex/build/sw/runtime)
set(VORTEX_INCLUDE_DIR /home/cs254A/vortex/sw/runtime/include)
set(VORTEX_POCL_LIB /root/tools/pocl/lib/libOpenCL.so.2)

ggml_add_backend_library(${TARGET_NAME}
    ggml-vortex.cpp
    ../../include/ggml-vortex.h
)

target_link_libraries(${TARGET_NAME} PRIVATE
    ${VORTEX_POCL_LIB}
    ${VORTEX_RUNTIME_DIR}/libvortex.so
    ggml-base
)
target_include_directories(${TARGET_NAME} PRIVATE
    ${VORTEX_INCLUDE_DIR}
    /usr/include
)
target_compile_definitions(${TARGET_NAME} PRIVATE GGML_USE_VORTEX)
message(STATUS "Vortex backend: using Vortex POCL at ${VORTEX_POCL_LIB}")