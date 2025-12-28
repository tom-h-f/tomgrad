#ifndef TOMGRAD_H
#define TOMGRAD_H

#include <time.h>
#include <stddef.h>
#include <stdlib.h>


#define SUCCESS 0
#define ERR_UNKNOWN 1
#define ERR_MEMORY_ALLOCATION 2
#define ERR_INVALID_BACKWARDS_OP 3

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
};

enum tg_backward_op {
    TG_BOP_MUL,
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


#define TENSOR_PRINT(tensor) tensor_print(tensor)
#define TENSOR_PRINT_GRADIENTS(tensor) tensor_print_grads(tensor)

tg_err_t tensor_init(size_t dims[], size_t n_dims, tg_tensor_t** ptr);
void tensor_free(tg_tensor_t* tensor);

tg_err_t tensor_scalar_add(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_scalar_sub(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_scalar_mul(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_scalar_div(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_sqrt(tg_tensor_t* tensor);
tg_err_t tensor_abs(tg_tensor_t* tensor);

void tensor_print(tg_tensor_t* tensor);
void tensor_print_grads(tg_tensor_t* tensor);
size_t tensor_total_elements(tg_tensor_t* tensor);
tg_err_t tensor_create_graph(tg_tensor_t* tensor, \
                             tg_tensor_t* a, \
                             tg_tensor_t* b, \
                             enum tg_backward_op op);


tg_tensor_t* tensor_mul(tg_tensor_t* a, tg_tensor_t* b);
tg_err_t tensor_backward_mul(tg_tensor_t* tensor);
tg_err_t  tensor_backward_sum(tg_tensor_t* tensor);


tg_err_t tensor_shape_init(size_t dims[], size_t n_dims, tg_tensor_shape_t* shape);

// Utility functions
size_t total_elements_for_dimensions(size_t dims[], size_t n_dims);

#endif // TOMGRAD_H
