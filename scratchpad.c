#include "../tomgrad/tomgrad.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

int main(void) {

    tg_tensor_t* a = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&a, 1.5, 2, 3, 10);
    tg_tensor_t* b = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&b, 2.25, 2, 3, 10);

    tensor_print(a);
    tensor_print(b);

    tg_tensor_t* tensor = tensor_mul(a, b);

    TENSOR_PRINT_GRADIENTS(tensor->input_tensors[0]);
    tensor->backward(tensor);
    TENSOR_PRINT_GRADIENTS(tensor->input_tensors[0]);

    printf("\n%lu\n", tensor->n_elements);

    return 0;
}
