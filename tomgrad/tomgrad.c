#include "tomgrad.h"
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


tg_err_t tensor_init(size_t dims[], size_t n_dims, tg_tensor_t** ptr) {
    assert(dims != NULL);
    assert(n_dims > 0);

    size_t tensor_size = total_elements_for_dimensions(dims, n_dims) * sizeof(tg_value_t);

    size_t total_size = (2 * tensor_size) + sizeof(tg_tensor_t);

    tg_tensor_t* tensor = calloc(1, total_size);
    if (!tensor) {return ERR_MEMORY_ALLOCATION; }
    tensor_shape_init(dims, n_dims, &tensor->shape);

    tensor->n_elements = tensor_total_elements(tensor);

    tensor->vals = (tg_value_t*)(tensor+1);
    tensor->grads = tensor->vals + tensor->n_elements;


    for(size_t i = 0; i< tensor->n_elements; i++) {
        tensor->grads[i] = 1.0;
    }
    *ptr = tensor;
    return SUCCESS;
}

void tensor_free(tg_tensor_t* tensor) {
    assert(tensor != NULL);
    for (size_t i = 0; i < tensor->n_input_tensors; ++i) {
        if(tensor->input_tensors[i]) {
            free(tensor->input_tensors[i]);
        }
    }
    free(tensor);
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
    for (size_t i = 0; i < (tensor)->n_elements; i++) { 
          (tensor)->vals[i] = sqrtf((tensor)->vals[i]); 
    }
    return SUCCESS;
}

tg_err_t tensor_abs(tg_tensor_t* tensor) {
    for (size_t i = 0; i < (tensor)->n_elements; i++) { 
          (tensor)->vals[i] = fabsf((tensor)->vals[i]); 
    }
    return SUCCESS;
}

void tensor_print(tg_tensor_t* tensor) {
    printf("Tensor {\n\t");
    for(size_t i = 0; i< tensor->n_elements; i++) {
        printf("[%0.03f] ", tensor->vals[i]);
        if (i != 0 && (i+1) % 3 == 0) { printf("\n\t");}
    }
    printf("\r}\n");
}

void tensor_print_grads(tg_tensor_t* tensor) {
    printf("Tensor Gradients {\n\t");
    for(size_t i = 0; i< tensor->n_elements; i++) {
        printf("[%0.03f] ", tensor->grads[i]);
        if (i != 0 && (i+1) % 3 == 0) { printf("\n\t");}
    }
    printf("\n}\n");
}

tg_err_t tensor_backward_mul(tg_tensor_t* tensor) {
    assert(tensor->input_tensors[0] != NULL);
    assert(tensor->input_tensors[1] != NULL);
    assert(tensor->n_input_tensors == 2);
    assert(tensor->grads != NULL);
    assert(tensor->input_tensors[0]->shape.n_dimensions == tensor->input_tensors[1]->shape.n_dimensions);

    tg_tensor_t* A = tensor->input_tensors[0];
    tg_tensor_t* B = tensor->input_tensors[1];

    for (size_t i = 0; i < tensor->n_elements; i++) {
        A->grads[i] += tensor->grads[i] * B->vals[i];
        B->grads[i] += tensor->grads[i] * A->vals[i];
    }

    return SUCCESS;
}

tg_err_t  tensor_backward_sum(tg_tensor_t* tensor) {
    assert(tensor->input_tensors[0] != NULL);
    assert(tensor->input_tensors[1] != NULL);
    return SUCCESS;
}

tg_tensor_t* tensor_mul(tg_tensor_t* a, tg_tensor_t* b) {
    tg_tensor_t* tensor = NULL;
    tensor_init(a->shape.dimensions, a->shape.n_dimensions, &tensor);

    for(size_t i = 0; i< tensor->n_elements; i++) {
        tensor->vals[i] = a->vals[i] * b->vals[i];
    }

    tensor_create_graph(tensor, a, b, TG_BOP_MUL);
    return tensor;
}

tg_err_t tensor_create_graph(tg_tensor_t* tensor, \
                             tg_tensor_t* a, \
                             tg_tensor_t* b, \
                             enum tg_backward_op op) {
    assert(tensor != NULL);
    assert(a != NULL);
    assert(b != NULL);

    tensor->input_tensors = calloc(2, sizeof(tg_tensor_t*));
    tensor->n_input_tensors = 2;
    tensor->input_tensors[0] = a;
    tensor->input_tensors[1] = b;

    switch (op) {
        case TG_BOP_MUL:
            tensor->backward = tensor_backward_mul;
            break;

        default:
            TENSOR_DESTROY(tensor);
            assert(false);
    }

    return SUCCESS;
}



tg_err_t tensor_shape_init(size_t dims[], size_t n_dims, tg_tensor_shape_t* shape) {
    assert(dims != NULL);
    assert(n_dims > 0);
    if (!shape) {return ERR_MEMORY_ALLOCATION; }

    shape->n_dimensions = n_dims;

    shape->dimensions = calloc(shape->n_dimensions, sizeof(size_t));
    memcpy((void*)shape->dimensions, (const void*) dims, sizeof(size_t)*n_dims);
    assert(*dims == *shape->dimensions);

    shape->strides = calloc(shape->n_dimensions, sizeof(size_t));
    for(size_t i = 0; i < shape->n_dimensions; i++) {
        size_t stride_length = 1;
        for (size_t j = i; j < shape->n_dimensions; j++) {
            stride_length *= dims[j];
        }
        shape->strides[i] = stride_length;
    }
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
