/*
** Copyright © 2018, Oracle and/or its affiliates. All rights reserved.
** Licensed under the Universal Permissive License v 1.0 as shown at http://oss.oracle.com/licenses/upl.
*/

#ifndef GP_CAFFE2
#define GP_CAFFE2

#ifdef __cplusplus 
extern "C" {
#endif

#include <stddef.h>


typedef struct c2_engine_ctx c2_engine_ctx;

c2_engine_ctx* c2_engine_create(int use_cuda);
int c2_engine_initialize_onnx(c2_engine_ctx *ctx, char *model_data, size_t len);
int c2_engine_initialize_caffe2(c2_engine_ctx *ctx, char *init_data, size_t init_data_len, char *pred_data, size_t pred_data_len);

int c2_engine_get_input_count(c2_engine_ctx * ctx);
const char *c2_engine_get_input_name(c2_engine_ctx * ctx, int i);
int c2_engine_get_output_count(c2_engine_ctx * ctx);
const char *c2_engine_get_output_name(c2_engine_ctx * ctx, int i);

int c2_engine_get_dimensions(c2_engine_ctx *ctx, char *name, int64_t *dimensions);
void c2_engine_register_input(c2_engine_ctx *ctx, char *name, int64_t *shape, int len, int dtype);

int c2_set_input_batch(c2_engine_ctx *ctx, char *name, void *input, int el_count, int64_t *shape, int shape_len);
int c2_execute_batch(c2_engine_ctx *ctx);

int c2_engine_get_output(c2_engine_ctx *ctx, int i, void *output, int64_t *shape, int shape_len);
int c2_engine_get_output_size(c2_engine_ctx *ctx, int i);

int c2_engine_get_dtype(c2_engine_ctx *ctx, char *name);
int c2_engine_get_itemsize(c2_engine_ctx *ctx, char *name);

int c2_engine_get_output_index(c2_engine_ctx *ctx, char *name);

enum TensorProto_DataType {
  TensorProto_DataType_UNDEFINED = 0,
  TensorProto_DataType_FLOAT = 1,
  TensorProto_DataType_INT32 = 2,
  TensorProto_DataType_BYTE = 3,
  TensorProto_DataType_STRING = 4,
  TensorProto_DataType_BOOL = 5,
  TensorProto_DataType_UINT8 = 6,
  TensorProto_DataType_INT8 = 7,
  TensorProto_DataType_UINT16 = 8,
  TensorProto_DataType_INT16 = 9,
  TensorProto_DataType_INT64 = 10,
  TensorProto_DataType_FLOAT16 = 12,
  TensorProto_DataType_DOUBLE = 13
};

#ifdef __cplusplus
}
#endif

#endif
