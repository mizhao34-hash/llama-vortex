#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

GGML_BACKEND_API ggml_backend_t ggml_backend_vortex_init(void);


GGML_BACKEND_API bool ggml_backend_is_vortex(ggml_backend_t backend);


GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_vortex_buffer_type(void);
GGML_BACKEND_API ggml_guid_t        ggml_backend_vortex_guid(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_vortex_reg(void);
#ifdef  __cplusplus
}
#endif