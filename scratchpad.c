#include "../tomgrad/tomgrad.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>


void train_simple_linear_model(void) {
    float learning_rate = 0.01f;
    size_t num_iterations = 500;

    // learn y = a*x such to minimize (a*x - target)^2
    // In simpler terms, learn a*x to be as close to `target` as possible. 
    // target = 2
    tg_tensor_t* target = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&target, 2.0f, 1);
    
    tg_tensor_t* a = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&a, 1.0f, 1);
    tg_tensor_t* w = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&w, 3.0f, 1);

    for(size_t i = 0; i < num_iterations; ++i) {
        TENSOR_GRADS_SET(a, 0.0);
        TENSOR_GRADS_SET(w, 0.0);
        auto pred = tensor_el_mul(a, w);
        auto y_hat = tensor_el_sub(pred, target);
        auto L = tensor_el_mul(y_hat, y_hat);
        tensor_backward_pass(L);
        for(size_t j = 0; j < w->n_elements; ++j) {
            w->vals[j] = w->vals[j] - learning_rate * w->grads[j];
        }
        printf("%.4f->%.4f\n", L->vals[0], w->vals[0]);
        tensor_free_recursive(L);
    }
    tensor_free(a);
    tensor_free(w);
}

void test_elementwise_ops(void) {
    tg_tensor_t* a = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&a, 1.5, 2, 3, 10);
    tg_tensor_t* b = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&b, 2.25, 2, 3, 10);

    tensor_print(a);
    tensor_print(b);

    tg_tensor_t* c = tensor_el_mul(a, b);
    UNWRAP(tensor_backward_pass(c));

    TENSOR_PRINT_GRADIENTS(a);
    TENSOR_PRINT_GRADIENTS(b);
    c->backward(c);
    TENSOR_PRINT_GRADIENTS(a);
    TENSOR_PRINT_GRADIENTS(b);
    TENSOR_PRINT_GRADIENTS(c);

    printf("\n%lu\n", c->n_elements);
}

int main(void) {
    train_simple_linear_model();
    return 0;
}
