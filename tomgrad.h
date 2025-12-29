#ifndef TOMGRAD_H
#define TOMGRAD_H

#include <time.h>
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>


// =======================================================
// DEFINITIONS [START]
// =======================================================


#define SUCCESS 0
#define ERR_UNKNOWN 1
#define ERR_MEMORY_ALLOCATION 2
#define ERR_INVALID_BACKWARDS_OP 3

#define TODO() assert(false && "TODO") 
#define UNREACHABLE() assert(false && "UNREACHABLE") 
#define UNWRAP(expr) do { \
		tg_err_t err = (expr); \
		if (err != SUCCESS) { \
            fprintf(stderr, "Panic at %s:%d - Error code %d\n", __FILE__, __LINE__, err); \
            abort(); \
		} \
} while (0)


typedef float tg_value_t;
typedef int tg_err_t;


typedef struct {
    size_t n_dimensions;
    size_t* dimensions;
    size_t* strides;
} tg_tensor_shape_t;

typedef struct tg_tensor_t tg_tensor_t;

struct tg_tensor_t {
    size_t n_elements;
    tg_tensor_shape_t shape;

    tg_value_t* vals;
    tg_value_t* grads;
    tg_err_t (*backward)(tg_tensor_t* self);

    tg_tensor_t** input_tensors;
    size_t n_input_tensors;
    size_t ref_count;
};

enum tg_backward_op {
    TG_BOP_EL_ADD,
    TG_BOP_EL_SUB,
    TG_BOP_EL_MUL,
    TG_BOP_EL_DIV,
    TG_BOP_MAT_MUL,
    TG_BOP_SUM_REDUCTION,
    TG_BOP_MEAN_REDUCTION,
};


#define TENSOR_SCALAR_OP(tensor, scalar, op) \
		do { \
				for (size_t i = 0; i < (tensor)->n_elements; i++) { \
						(tensor)->vals[i] = (tensor)->vals[i] op (scalar); \
				} \
		} while (0)

#define TENSOR_CREATE(tensor_ptr, ...) \
		do { \
				size_t dims[] = {__VA_ARGS__}; \
				UNWRAP(tensor_init(dims, sizeof(dims)/sizeof(dims[0]), tensor_ptr)); \
		} while (0)

#define TENSOR_DESTROY(tensor) \
		do { \
				if (tensor) { \
						free(tensor); \
						tensor = NULL; \
				} \
		} while (0)

#define TENSOR_CREATE_ZEROS(tensor_ptr, ...) \
		TENSOR_CREATE(tensor_ptr, __VA_ARGS__)

#define TENSOR_CREATE_ONES(tensor_ptr, ...) \
		do { \
				TENSOR_CREATE(tensor_ptr, __VA_ARGS__); \
				for (size_t i = 0; i < (*tensor_ptr)->n_elements; i++) { \
						(*tensor_ptr)->vals[i] = 1.0; \
				} \
		} while (0)

#define TENSOR_CREATE_FILLED(tensor_ptr, value, ...) \
		do { \
				TENSOR_CREATE(tensor_ptr, __VA_ARGS__); \
				for (size_t i = 0; i < (*tensor_ptr)->n_elements; i++) { \
						(*tensor_ptr)->vals[i] = (value); \
				} \
		} while (0)

#define TENSOR_CREATE_RANDOM(tensor_ptr, ...) \
		do { \
                srand(time(NULL)); \
				TENSOR_CREATE(tensor_ptr, __VA_ARGS__); \
				for (size_t i = 0; i < (*tensor_ptr)->n_elements; i++) { \
						(*tensor_ptr)->vals[i] = (float)rand() - (float)rand(); \
				} \
		} while (0)

#define TENSOR_CREATE_RANGE(tensor_ptr, start, step, ...) \
		do { \
				TENSOR_CREATE(tensor_ptr, __VA_ARGS__); \
				tg_value_t current = (start); \
				for (size_t i = 0; i < (*tensor_ptr)->n_elements; i++) { \
						(*tensor_ptr)->vals[i] = current; \
						current += (step); \
				} \
		} while (0)

#define TENSOR_FOR_EACH(tensor, var, code) \
		for (size_t i = 0; i < (tensor)->n_elements; i++) { \
				tg_value_t var = (tensor)->vals[i]; \
				code; \
		}

#define TENSOR_ASSERT_EQUAL(t1, t2) \
		do { \
				assert((t1)->n_elements == (t2)->n_elements); \
				for (size_t i = 0; i < (t1)->n_elements; i++) { \
						assert((t1)->vals[i] == (t2)->vals[i]); \
				} \
		} while (0)

#define TENSOR_GRADS_SET(t, v) \
		do { \
				for (size_t i = 0; i < t->n_elements; i++) { \
                     t->grads[i] = v; \
				} \
		} while (0)


#define TENSOR_PRINT(tensor) tensor_print(tensor)
#define TENSOR_PRINT_GRADIENTS(tensor) tensor_print_grads(tensor)

tg_err_t tensor_init(size_t dims[], size_t n_dims, tg_tensor_t** ptr);
void tensor_free(tg_tensor_t* tensor);
void tensor_free_recursive(tg_tensor_t* tensor);

tg_err_t tensor_scalar_add(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_scalar_sub(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_scalar_mul(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_scalar_div(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_sqrt(tg_tensor_t* tensor);
tg_err_t tensor_abs(tg_tensor_t* tensor);

tg_value_t tensor_dot_product(tg_tensor_t* a, tg_tensor_t* b);

void tensor_print(tg_tensor_t* tensor);
void tensor_print_grads(tg_tensor_t* tensor);
void tensor_shape_print(tg_tensor_shape_t* shape);
size_t tensor_total_elements(tg_tensor_t* tensor);
tg_err_t tensor_create_graph(tg_tensor_t* tensor, \
                             tg_tensor_t* a, \
                             tg_tensor_t* b, \
                             enum tg_backward_op op);


tg_tensor_t* tensor_el_add(tg_tensor_t* a, tg_tensor_t* b);
tg_tensor_t* tensor_el_sub(tg_tensor_t* a, tg_tensor_t* b);
tg_tensor_t* tensor_el_mul(tg_tensor_t* a, tg_tensor_t* b);
tg_tensor_t* tensor_el_div(tg_tensor_t* a, tg_tensor_t* b);
tg_err_t tensor_backward_el_add(tg_tensor_t* tensor);
tg_err_t tensor_backward_el_sub(tg_tensor_t* tensor);
tg_err_t tensor_backward_el_mul(tg_tensor_t* tensor);
tg_err_t tensor_backward_el_div(tg_tensor_t* tensor);

/*
* tg_err_t tensor_backward_mat_mul(tg_tensor_t* tensor);
*/


tg_err_t tensor_shape_init(size_t dims[], size_t n_dims, tg_tensor_shape_t* shape);
void tensor_shape_print(tg_tensor_shape_t* shape);

// Utility functions
size_t total_elements_for_dimensions(size_t dims[], size_t n_dims);


// =======================================================
// DEFINITIONS [END]
// =======================================================



tg_err_t tensor_init(size_t dims[], size_t n_dims, tg_tensor_t** ptr) {
    assert(dims != NULL);
    assert(n_dims > 0);

    size_t tensor_size = total_elements_for_dimensions(dims, n_dims) * sizeof(tg_value_t);
    size_t total_size = (2 * tensor_size) + sizeof(tg_tensor_t);

    tg_tensor_t* tensor = calloc(1, total_size);
    if (!tensor) {return ERR_MEMORY_ALLOCATION; }
    tensor_shape_init(dims, n_dims, &tensor->shape);

    tensor->ref_count = 1;
    tensor->n_elements = tensor_total_elements(tensor);

    tensor->vals = (tg_value_t*)(tensor+1);
    tensor->grads = (tg_value_t*)(tensor->vals + tensor->n_elements);

    *ptr = tensor;
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
        for (size_t j = i + 1; j < shape->n_dimensions; j++) {
            stride_length *= dims[j];
        }
        shape->strides[i] = stride_length;
    }
    return SUCCESS;
}

void tensor_free(tg_tensor_t* tensor) {
    assert(tensor != NULL);
    free(tensor->input_tensors);
    free(tensor);
}

void tensor_free_recursive(tg_tensor_t* tensor) {
    assert(tensor != NULL);

    if(tensor->ref_count > 1) {
        tensor->ref_count -= 1;
        return;
    }

    for(size_t i = 0; i < tensor->n_input_tensors; ++i) {
        tensor_free_recursive(tensor->input_tensors[i]);
    }
    free(tensor->input_tensors);
    free(tensor);
}

tg_err_t tensor_backward_pass(tg_tensor_t* tensor) {
    TENSOR_GRADS_SET(tensor, 1.0);
    return tensor->backward(tensor);
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

tg_err_t tensor_create_graph(tg_tensor_t* tensor, \
                              tg_tensor_t* a, \
                              tg_tensor_t* b, \
                              enum tg_backward_op op) {
    assert(tensor != NULL);
    assert(a != NULL);
    assert(b != NULL);

    // During backward pass:
    //   - Calling backward(C) computes dL/dA and dL/dB
    //   - Then recursively calls backward(A) and backward(B)
    //
    // This allows the loss gradient to flow backward through the entire graph:
    //   Loss -> ... -> C -> A, B -> ... -> parameters
    //
    tensor->n_input_tensors = 2;
    tensor->input_tensors = calloc(tensor->n_input_tensors, \
                                   sizeof(tg_tensor_t*));
    tensor->input_tensors[0] = a;
    tensor->input_tensors[1] = b;

    a->ref_count += 1;
    b->ref_count += 1;

    switch (op) {
        case TG_BOP_EL_ADD:
            // Addition:
            // d(A+B)/dA = 1,  d(A+B)/dB = 1
            tensor->backward = tensor_backward_el_add;
            break;
        case TG_BOP_EL_SUB:
            // Subtraction:
            // d(A-B)/dA = 1,  d(A-B)/dB = -1
            tensor->backward = tensor_backward_el_sub;
            break;
        case TG_BOP_EL_MUL:
            // Multiplication:
            // d(A*B)/dA = B,  d(A*B)/dB = A
            tensor->backward = tensor_backward_el_mul;
            break;
        case TG_BOP_EL_DIV:
            // Division:
            // d(A/B)/dA = 1/B, d(A/B)/dB = -A/B²
            tensor->backward = tensor_backward_el_div;
            break;
        case TG_BOP_MAT_MUL:
        case TG_BOP_MEAN_REDUCTION:
        case TG_BOP_SUM_REDUCTION:
            TODO();
        default:
            TENSOR_DESTROY(tensor);
            UNREACHABLE();
    }

    return SUCCESS;
}

tg_err_t  tensor_backward_el_add(tg_tensor_t* tensor) {
    if(tensor->n_input_tensors == 0) {
        return SUCCESS;
    }

    assert(tensor->n_input_tensors == 2);
    assert(tensor->input_tensors[0] != NULL);
    assert(tensor->input_tensors[1] != NULL);
    assert(tensor->input_tensors[0]->shape.n_dimensions\
           == tensor->input_tensors[1]->shape.n_dimensions);

    tg_tensor_t* A = tensor->input_tensors[0];
    tg_tensor_t* B = tensor->input_tensors[1];

    for (size_t i = 0; i < tensor->n_elements; i++) {
         A->grads[i] += tensor->grads[i];
         B->grads[i] += tensor->grads[i];
    }
    
    if(A->backward) A->backward(A);
    if(B->backward) B->backward(B);


    return SUCCESS;
}

tg_err_t tensor_backward_el_sub(tg_tensor_t* tensor) {
    if(tensor->n_input_tensors == 0) {
        return SUCCESS;
    }
    assert(tensor->n_input_tensors == 2);
    assert(tensor->input_tensors[0] != NULL);
    assert(tensor->input_tensors[1] != NULL);
    assert(tensor->input_tensors[0]->shape.n_dimensions\
           == tensor->input_tensors[1]->shape.n_dimensions);

    tg_tensor_t* A = tensor->input_tensors[0];
    tg_tensor_t* B = tensor->input_tensors[1];

    for (size_t i = 0; i < tensor->n_elements; i++) {
         A->grads[i] += tensor->grads[i];
         B->grads[i] -= tensor->grads[i];
    }
    
    if(A->backward) A->backward(A);
    if(B->backward) B->backward(B);

    return SUCCESS;
}

tg_err_t tensor_backward_el_mul(tg_tensor_t* tensor) {
    assert(tensor->grads != NULL);

    if(tensor->n_input_tensors == 0) {
        return SUCCESS;
    }

    assert(tensor->n_input_tensors == 2);
    assert(tensor->input_tensors[0] != NULL);
    assert(tensor->input_tensors[1] != NULL);
    assert(tensor->input_tensors[0]->shape.n_dimensions \
           == tensor->input_tensors[1]->shape.n_dimensions);

    tg_tensor_t* A = tensor->input_tensors[0];
    tg_tensor_t* B = tensor->input_tensors[1];

    for (size_t i = 0; i < tensor->n_elements; i++) {
        A->grads[i] += tensor->grads[i] * B->vals[i];
        B->grads[i] += tensor->grads[i] * A->vals[i];
    }
    
    if(A->backward) A->backward(A);
    if(B->backward) B->backward(B);

    return SUCCESS;
}

tg_err_t tensor_backward_el_div(tg_tensor_t* tensor) {
    assert(tensor->grads != NULL);

    if(tensor->n_input_tensors == 0) {
        return SUCCESS;
    }

    assert(tensor->n_input_tensors == 2);
    assert(tensor->input_tensors[0] != NULL);
    assert(tensor->input_tensors[1] != NULL);
    assert(tensor->input_tensors[0]->shape.n_dimensions \
           == tensor->input_tensors[1]->shape.n_dimensions);

    tg_tensor_t* A = tensor->input_tensors[0];
    tg_tensor_t* B = tensor->input_tensors[1];

    for (size_t i = 0; i < tensor->n_elements; i++) {
        A->grads[i] += tensor->grads[i] * (1/B->vals[i]);
        B->grads[i] += tensor->grads[i] * ((-1 * A->vals[i]) / (B->vals[i] * B->vals[i]));
    }
    
    if(A->backward) A->backward(A);
    if(B->backward) B->backward(B);

    return SUCCESS;
}

tg_tensor_t* tensor_el_add(tg_tensor_t* a, tg_tensor_t* b) {
    tg_tensor_t* tensor = NULL;
    UNWRAP(tensor_init(a->shape.dimensions, a->shape.n_dimensions, &tensor));

    for(size_t i = 0; i< tensor->n_elements; i++) {
        tensor->vals[i] = a->vals[i] + b->vals[i];
    }

    tensor_create_graph(tensor, a, b, TG_BOP_EL_ADD);
    return tensor;
}

tg_tensor_t* tensor_el_sub(tg_tensor_t* a, tg_tensor_t* b) {
    tg_tensor_t* tensor = NULL;
    UNWRAP(tensor_init(a->shape.dimensions, a->shape.n_dimensions, &tensor));

    for(size_t i = 0; i< tensor->n_elements; i++) {
        tensor->vals[i] = a->vals[i] - b->vals[i];
    }

    tensor_create_graph(tensor, a, b, TG_BOP_EL_SUB);
    return tensor;
}

tg_tensor_t* tensor_el_mul(tg_tensor_t* a, tg_tensor_t* b) {
    tg_tensor_t* tensor = NULL;
    UNWRAP(tensor_init(a->shape.dimensions, a->shape.n_dimensions, &tensor));

    for(size_t i = 0; i< tensor->n_elements; i++) {
        tensor->vals[i] = a->vals[i] * b->vals[i];
    }

    tensor_create_graph(tensor, a, b, TG_BOP_EL_MUL);
    return tensor;
}

tg_tensor_t* tensor_el_div(tg_tensor_t* a, tg_tensor_t* b) {
    tg_tensor_t* tensor = NULL;
    UNWRAP(tensor_init(a->shape.dimensions, a->shape.n_dimensions, &tensor));

    for(size_t i = 0; i< tensor->n_elements; i++) {
        tensor->vals[i] = a->vals[i] / b->vals[i];
    }

    tensor_create_graph(tensor, a, b, TG_BOP_EL_DIV);
    return tensor;
}

float tensor_dot_product(tg_tensor_t* a, tg_tensor_t* b) {
    assert(a != NULL);
    assert(b != NULL);
    assert(a->vals != NULL);
    assert(b->vals != NULL);
    assert(a->n_elements == b->n_elements);

    tg_value_t result = 0.0f;
    for(size_t i = 0; i < a->n_elements; ++i) {
        result += a->vals[i] * b->vals[i];
    }
    return result;
}









// ==============================
//            Utils
// ==============================
void tensor_print(tg_tensor_t* tensor) {

    if(tensor->n_elements == 0) {
        printf("Tensor {\n}\n");
        return;
    }

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

void tensor_shape_print(tg_tensor_shape_t* shape) {
    printf("Tensor Shape {\n");
    printf("  n_dimensions: %zu,\n", shape->n_dimensions);
    printf("  dimensions: [");
    for (size_t i = 0; i < shape->n_dimensions; i++) {
        printf("%zu", shape->dimensions[i]);
        if (i < shape->n_dimensions - 1) printf(", ");
    }
    printf("],\n");
    printf("  strides: [");
    for (size_t i = 0; i < shape->n_dimensions; i++) {
        printf("%zu", shape->strides[i]);
        if (i < shape->n_dimensions - 1) printf(", ");
    }
    printf("]\n");
    printf("}\n");
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

#endif // TOMGRAD_H
