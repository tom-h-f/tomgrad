#include "tomgrad.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


tg_err_t tensor_init(size_t dims[], size_t n_dims, tg_tensor_t** ptr) {
    assert(dims != NULL);
    assert(n_dims > 0);


    size_t total_size = \
        (total_elements_for_dimensions(dims, n_dims) * \
        sizeof(tg_value_t)) + \
        sizeof(tg_tensor_t);

    tg_tensor_t* tensor = calloc(1, total_size);
    if (!tensor) {return ERR_MEMORY_ALLOCATION; }
    tensor_shape_init(dims, n_dims, &tensor->shape);

    tensor->size = tensor_total_elements(tensor);
    tensor->vals = (tg_value_t*)(tensor+1);

    assert(tensor->size == tensor_total_elements(tensor));
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
          (tensor)->vals[i] = sqrtf((tensor)->vals[i]); 
    }
    return SUCCESS;
}

tg_err_t tensor_abs(tg_tensor_t* tensor) {
    for (size_t i = 0; i < (tensor)->size; i++) { 
          (tensor)->vals[i] = fabsf((tensor)->vals[i]); 
    }
    return SUCCESS;
}

void tensor_print(tg_tensor_t* tensor) {
    for(size_t i = 0; i< tensor->size; i++) {
        printf("[%09.03f] ", tensor->vals[i]);
        if (i != 0 && (i+1) % 3 == 0) { printf("\n");}
    }
}


tg_err_t tensor_shape_init(size_t dims[], size_t n_dims, tg_tensor_shape_t* shape) {
    assert(dims != NULL);
    assert(n_dims > 0);
    if (!shape) {return ERR_MEMORY_ALLOCATION; }

    shape->n_dimensions = n_dims;
    shape->dimensions = dims;

    assert(&shape->dimensions != &dims);
    assert(*dims == *shape->dimensions);

    // TODO: Calculate strides
    return SUCCESS;
}

size_t tensor_total_elements(tg_tensor_t* tensor) {
    assert(tensor->shape.n_dimensions > 0);

    size_t n = 1;
    for(size_t i = 0; i < tensor->shape.n_dimensions; i++) {
        n *= tensor->shape.dimensions[i];
    }

    total_elements_for_dimensions(tensor->shape.dimensions, tensor->shape.n_dimensions);
    return n;
}

size_t total_elements_for_dimensions(size_t dims[], size_t n_dims) {
    size_t n = 1;
    for(size_t i = 0; i < n_dims; i++) {
        n *= dims[i];
    }
    return n;
}
