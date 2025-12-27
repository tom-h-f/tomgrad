#ifndef TOMGRAD_H
#define TOMGRAD_H

#include <math.h>
#include <stddef.h>

typedef long double tg_value_t;
typedef int tg_err_t;

#define SUCCESS 0
#define ERROR_UNKNOWN 1


#define UNWRAP(expr) do { \
		tg_err_t _err = (expr); \
		if (_err != SUCCESS) { \
				panic(_err); \
		} \
} while (0)




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

#define TENSOR_MAP(tensor, expr) \
		do { \
				for (size_t i = 0; i < (tensor)->size; i++) { \
						(tensor)->vals[i] = (expr); \
				} \
		} while (0)


tg_err_t tensor_init(int dims[], size_t n_dims, tg_tensor_t** ptr);
tg_err_t tensor_scalar_add(tg_tensor_t* tensor, tg_value_t scalar);
void tensor_print(tg_tensor_t* tensor);



void panic(tg_err_t err);
#endif // TOMGRAD_H
