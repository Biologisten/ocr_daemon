#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void *ocr_create(const char *model_dir, char **error);
char *ocr_run(void *engine, const unsigned char *data, size_t size, char **error);
void ocr_destroy(void *engine);
void ocr_free(void *value);

#ifdef __cplusplus
}
#endif
