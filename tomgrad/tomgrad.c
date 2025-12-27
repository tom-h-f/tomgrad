#include "tomgrad.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


tg_err_t tensor_init(int dims[], size_t n_dims, tg_tensor_t** ptr) {
		tg_tensor_t* tensor = (tg_tensor_t*)malloc(sizeof(tg_tensor_t));

		size_t n_values = 0;
		for(size_t i = 0; i < n_dims; i++) {
				n_values += dims[i];
		}
		tensor->size = n_values;
		tensor->vals = (tg_value_t*)calloc(n_values, sizeof(tg_value_t));

#if DEBUG
		assert(tensor->size = n_values);
		for(size_t i = 0; i< n_values; i++) {
				assert(tensor->vals[i] == 0);
		}
#endif

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




void tensor_print(tg_tensor_t* tensor) {
		for(size_t i = 0; i< tensor->size; i++) {
				printf("[%.6Lf] ", tensor->vals[i]);
				if (i != 0 && (i+1) % 6 == 0) { printf("\n");}
		}
}

void panic(tg_err_t err) {
		fprintf(stderr, "Panic: Error code %d\n", err);
		abort();
}
