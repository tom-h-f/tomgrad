#ifndef TOMGRAD_H
#define TOMGRAD_H

#include <math.h>
#include <stddef.h>
#include <stdlib.h>

typedef long double tg_value_t;
typedef int tg_err_t;

#define SUCCESS 0
#define ERR_UNKNOWN 1
#define ERR_MEMORY_ALLOCATION 2


#define UNWRAP(expr) do { \
		tg_err_t err = (expr); \
		if (err != SUCCESS) { \
            fprintf(stderr, "Panic: Error code %d\n", err); \
            abort(); \
		} \
} while (0)



typedef struct {
    size_t n_dimensions;
    size_t* dimensions;
    size_t* strides;
} tg_tensor_shape_t;


typedef struct {
		tg_value_t* vals;
		size_t size;
} tg_tensor_t;


#define TENSOR_SCALAR_OP(tensor, scalar, op) \
		do { \
				for (size_t i = 0; i < (tensor)->size; i++) { \
						(tensor)->vals[i] = (tensor)->vals[i] op (scalar); \
				} \
		} while (0)

#define TENSOR_CREATE(tensor_ptr, ...) \
		do { \
				size_t dims[] = {__VA_ARGS__}; \
				tg_tensor_shape_t* shape = NULL; \
				UNWRAP(tensor_shape_init(dims, sizeof(dims)/sizeof(dims[0]), &shape)); \
				UNWRAP(tensor_init(shape, tensor_ptr)); \
				tensor_shape_free(shape); \
		} while (0)


tg_err_t tensor_init(tg_tensor_shape_t* shape, tg_tensor_t** ptr);
tg_err_t tensor_scalar_add(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_scalar_sub(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_scalar_mul(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_scalar_div(tg_tensor_t* tensor, tg_value_t scalar);
tg_err_t tensor_sqrt(tg_tensor_t* tensor);

tg_err_t tensor_free(tg_tensor_t* tensor);
void tensor_print(tg_tensor_t* tensor);

tg_err_t tensor_shape_init(size_t dims[], size_t n_dims, tg_tensor_shape_t** ptr);

size_t tensor_shape_total_elements(tg_tensor_shape_t* shape);
tg_err_t tensor_shape_free(tg_tensor_shape_t* shape);

#endif // TOMGRAD_H
