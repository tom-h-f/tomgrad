#include "tomgrad.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


tg_err_t tensor_init(tg_tensor_shape_t* shape, tg_tensor_t** ptr) {
    assert(shape != NULL);
    assert(shape->dimensions != NULL);
    assert(shape->n_dimensions > 0);

    tg_tensor_t* tensor = (tg_tensor_t*)calloc(1, sizeof(tg_tensor_t));
    if (!tensor) {return ERR_MEMORY_ALLOCATION; }

    tensor->size = tensor_shape_total_elements(shape);
    tensor->vals = (tg_value_t*)calloc(tensor->size, sizeof(tg_value_t));

    assert(tensor->size = tensor_shape_total_elements(shape));
    for(size_t i = 0; i< tensor->size; i++) {
        assert(tensor->vals[i] == 0);
    }

    *ptr = tensor;
    return SUCCESS;
}

tg_err_t tensor_scalar_add(tg_tensor_t* tensor, tg_value_t scalar) {
      TENSOR_SCALAR_OP(tensor, scalar, +);
      return SUCCESS;
}
tg_err_t tensor_scalar_sub(tg_tensor_t* tensor, tg_value_t scalar) {
      TENSOR_SCALAR_OP(tensor, scalar, -);
      return SUCCESS;
}
tg_err_t tensor_scalar_mul(tg_tensor_t* tensor, tg_value_t scalar) {
      TENSOR_SCALAR_OP(tensor, scalar, *);
      return SUCCESS;
}
tg_err_t tensor_scalar_div(tg_tensor_t* tensor, tg_value_t scalar) {
      TENSOR_SCALAR_OP(tensor, scalar, /);
      return SUCCESS;
}

tg_err_t tensor_sqrt(tg_tensor_t* tensor) {
    for (size_t i = 0; i < (tensor)->size; i++) { 
          (tensor)->vals[i] = sqrtl((tensor)->vals[i]); 
    }
    return SUCCESS;
}

tg_err_t  tensor_free(tg_tensor_t* tensor) {
    assert(tensor != NULL);
    free(tensor->vals);
    free(tensor);
    return SUCCESS;
}


void tensor_print(tg_tensor_t* tensor) {
    for(size_t i = 0; i< tensor->size; i++) {
        printf("[%.6Lf] ", tensor->vals[i]);
        if (i != 0 && (i+1) % 6 == 0) { printf("\n");}
    }
}


tg_err_t tensor_shape_init(size_t dims[], size_t n_dims, tg_tensor_shape_t** ptr) {
    assert(dims != NULL);
    assert(n_dims > 0);

    tg_tensor_shape_t* shape = (tg_tensor_shape_t*)calloc(1, sizeof(tg_tensor_shape_t));
    if (!shape) {return ERR_MEMORY_ALLOCATION; }

    shape->n_dimensions = n_dims;
    shape->dimensions = (size_t*)calloc(n_dims, sizeof(size_t));
    memcpy((void*)shape->dimensions, \
           (const void*)dims, \
           n_dims*sizeof(size_t));

    assert(*dims == *shape->dimensions);

    // TODO: Calculate strides

    *ptr = shape;
    return SUCCESS;
}

size_t tensor_shape_total_elements(tg_tensor_shape_t* shape) {
    assert(shape != NULL);
    assert(shape->n_dimensions > 0);

    size_t n = 1;
    for(size_t i = 0; i < shape->n_dimensions; i++) {
        n *= shape->dimensions[i];
    }
    return n;
}




tg_err_t tensor_shape_free(tg_tensor_shape_t* shape) {
    free(shape->dimensions);
    free(shape->strides);
    free(shape);
    return SUCCESS;
}
