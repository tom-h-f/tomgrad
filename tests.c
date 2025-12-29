#include "unity/unity_internals.h"
#include "unity/unity.h"
#include "tomgrad.h"

void setUp(void) {}
void tearDown(void) {}

void test_tensor_init_creates_tensor(void) {
    tg_tensor_t* tensor = (tg_tensor_t*)NULL;
    TENSOR_CREATE(&tensor, 3);
    TEST_ASSERT_NOT_NULL(tensor);
    TEST_ASSERT_EQUAL(3, tensor->n_elements);
    TEST_ASSERT_NOT_NULL(tensor->vals);
    tensor_free(tensor);
}

void test_tensor_init_multiple_dims(void) {
    tg_tensor_t* tensor = (tg_tensor_t*)NULL;
    TENSOR_CREATE(&tensor, 2, 3);
    TEST_ASSERT_EQUAL(6, tensor->n_elements);
    TEST_ASSERT_NOT_NULL(tensor->vals);
    tensor_free(tensor);
}


void test_tensor_scalar_add(void) {
    tg_tensor_t* tensor = (tg_tensor_t*)NULL;
    TENSOR_CREATE(&tensor, 3);
    tensor->vals[0] = 1.0;
    tensor->vals[1] = 2.0;
    tensor->vals[2] = 3.0;

    tensor_scalar_add(tensor, 5.0);

    TEST_ASSERT_EQUAL_DOUBLE(6.0, tensor->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(7.0, tensor->vals[1]);
    TEST_ASSERT_EQUAL_DOUBLE(8.0, tensor->vals[2]);
    tensor_free(tensor);
}

void test_tensor_scalar_add_negative(void) {
    tg_tensor_t* tensor = (tg_tensor_t*)NULL;
    TENSOR_CREATE(&tensor, 2);
    tensor->vals[0] = 10.0;
    tensor->vals[1] = 5.0;

    tensor_scalar_add(tensor, -3.0);

    TEST_ASSERT_EQUAL_DOUBLE(7.0, tensor->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(2.0, tensor->vals[1]);
    tensor_free(tensor);
}

void test_tensor_sqrt_all(void) {
    tg_tensor_t* tensor = (tg_tensor_t*)NULL;
    TENSOR_CREATE(&tensor, 2);
    UNWRAP(tensor_scalar_add(tensor, 64.0));
    UNWRAP(tensor_sqrt(tensor));

    for(size_t i = 0; i< tensor->n_elements; i++) {
        TEST_ASSERT_EQUAL_DOUBLE(8.0, tensor->vals[i]);
    }

    TEST_ASSERT_EQUAL_DOUBLE(8.0, tensor->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(8.0, tensor->vals[1]);
    tensor_free(tensor);
}


void test_tensor_shape_init_creates_shape(void) {
    size_t dims[] = {3, 3, 3};
    tg_tensor_t* tensor = NULL;
    TENSOR_CREATE(&tensor, 3, 3, 3);

    TEST_ASSERT_NOT_NULL(tensor);

    TEST_ASSERT_EQUAL(3, tensor->shape.n_dimensions);

    for(size_t i = 0; i< tensor->shape.n_dimensions; i++) {
        TEST_ASSERT_EQUAL_size_t(dims[i], tensor->shape.dimensions[i]);
    }

    TEST_ASSERT_EQUAL(3, tensor->shape.n_dimensions);

    tensor_free(tensor);
}

void test_tensor_shape_total_elements_calculation(void) {
    tg_tensor_t* tensor = (tg_tensor_t*)NULL;
    TENSOR_CREATE(&tensor, 3, 3, 3);
    TEST_ASSERT_EQUAL(27, tensor_total_elements(tensor));
    tensor_free(tensor);
}


void test_tensor_dot_product_simple_calculation(void) {
    tg_tensor_t* a = (tg_tensor_t*)NULL;
    tg_tensor_t* b = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&a, 2.0, 2);
    TENSOR_CREATE_FILLED(&b, 2.0, 2);

    tg_value_t result = tensor_dot_product(a, b);
    TEST_ASSERT_EQUAL(result, 8.0);

    tensor_free(a);
    tensor_free(b);
}

void test_backward_simple_multiplication(void) {
    tg_tensor_t* A = (tg_tensor_t*)NULL;
    tg_tensor_t* B = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&A, 2.0, 2);
    TENSOR_CREATE_FILLED(&B, 4.0, 2);
    A->vals[1] = 3.0;
    B->vals[1] = 5.0;

    tg_tensor_t* L = tensor_el_mul(A, B);

    TEST_ASSERT_EQUAL_DOUBLE(8.0, L->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(15.0, L->vals[1]);

    TENSOR_GRADS_SET(L, 1.0);
    L->backward(L);

    TEST_ASSERT_EQUAL_DOUBLE(4.0, A->grads[0]);
    TEST_ASSERT_EQUAL_DOUBLE(5.0, A->grads[1]); 
    TEST_ASSERT_EQUAL_DOUBLE(2.0, B->grads[0]);
    TEST_ASSERT_EQUAL_DOUBLE(3.0, B->grads[1]);

    tensor_free_recursive(L);
}


void test_backward_leaf_tensor_no_op(void) {
    tg_tensor_t* A = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&A, 2.0, 1);

    TEST_ASSERT_EQUAL(0, A->n_input_tensors);
    TEST_ASSERT_NULL(A->backward);

    tensor_free(A);
}

void test_reference_counting_prevents_double_free(void) {
    tg_tensor_t* A = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&A, 2.0, 1);
    tg_tensor_t* B = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&B, 3.0, 1);
    tg_tensor_t* E = NULL;
    TENSOR_CREATE_FILLED(&E, 5.0, 1);

    tg_tensor_t* C = tensor_el_mul(A, B);
    tg_tensor_t* D = tensor_el_mul(A, E);
    tg_tensor_t* L = tensor_el_mul(C, D);

    TEST_ASSERT_EQUAL(3, A->ref_count); 
    TEST_ASSERT_EQUAL(2, E->ref_count); 
    TEST_ASSERT_EQUAL(2, C->ref_count);
    TEST_ASSERT_EQUAL(2, D->ref_count);
    TEST_ASSERT_EQUAL(1, L->ref_count);

    if(L->backward) {
        L->backward(L);
    }

    tensor_free_recursive(L);

}

void test_tensor_free_recursive_complex_dag(void) {
    tg_tensor_t* t1 = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&t1, 1.0, 1);
    tg_tensor_t* t2 = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&t2, 2.0, 1);
    tg_tensor_t* t3 = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&t3, 3.0, 1);
    tg_tensor_t* t4 = (tg_tensor_t*)NULL;
    TENSOR_CREATE_FILLED(&t4, 4.0, 1);

    tg_tensor_t* t5 = tensor_el_mul(t1, t2);
    tg_tensor_t* t6 = tensor_el_mul(t2, t3);
    tg_tensor_t* t7 = tensor_el_mul(t3, t4);

    tg_tensor_t* t8 = tensor_el_mul(t5, t6);
    tg_tensor_t* t9 = tensor_el_mul(t6, t7);

    tg_tensor_t* t10 = tensor_el_mul(t8, t9);
    tg_tensor_t* t11 = tensor_el_mul(t9, t10);
    tg_tensor_t* t12 = tensor_el_mul(t10, t11);
    tg_tensor_t* L = tensor_el_mul(t11, t12);

    TEST_ASSERT_EQUAL(3, t2->ref_count); 
    TEST_ASSERT_EQUAL(3, t6->ref_count); 
    TEST_ASSERT_EQUAL(3, t9->ref_count); 
    TEST_ASSERT_EQUAL(3, t10->ref_count);
    TEST_ASSERT_EQUAL(3, t11->ref_count); 

    TEST_ASSERT_EQUAL(2, t1->ref_count);
    TEST_ASSERT_EQUAL(3, t3->ref_count);
    TEST_ASSERT_EQUAL(2, t4->ref_count);
    TEST_ASSERT_EQUAL(2, t5->ref_count);
    TEST_ASSERT_EQUAL(2, t7->ref_count);
    TEST_ASSERT_EQUAL(2, t8->ref_count);
    TEST_ASSERT_EQUAL(2, t12->ref_count); 
    TEST_ASSERT_EQUAL(1, L->ref_count);

    TEST_ASSERT_NOT_NULL(L);

    if(L->backward) {
        L->backward(L);
    }

    tensor_free_recursive(L);
}


void test_forward_el_add(void) {
    tg_tensor_t* A = NULL;
    tg_tensor_t* B = NULL;
    TENSOR_CREATE_FILLED(&A, 2.0, 2);
    TENSOR_CREATE_FILLED(&B, 3.0, 2);
    A->vals[1] = 5.0;
    B->vals[1] = 7.0;

    tg_tensor_t* C = tensor_el_add(A, B);

    TEST_ASSERT_EQUAL_DOUBLE(5.0, C->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(12.0, C->vals[1]);

    tensor_free_recursive(C);
}

void test_forward_el_sub(void) {
    tg_tensor_t* A = NULL;
    tg_tensor_t* B = NULL;
    TENSOR_CREATE_FILLED(&A, 8.0, 2);
    TENSOR_CREATE_FILLED(&B, 3.0, 2);
    A->vals[1] = 10.0;
    B->vals[1] = 4.0;

    tg_tensor_t* C = tensor_el_sub(A, B);

    TEST_ASSERT_EQUAL_DOUBLE(5.0, C->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(6.0, C->vals[1]);

    tensor_free_recursive(C);
}

void test_forward_el_div(void) {
    tg_tensor_t* A = NULL;
    tg_tensor_t* B = NULL;
    TENSOR_CREATE_FILLED(&A, 8.0, 2);
    TENSOR_CREATE_FILLED(&B, 2.0, 2);
    A->vals[1] = 12.0;
    B->vals[1] = 4.0;

    tg_tensor_t* C = tensor_el_div(A, B);

    TEST_ASSERT_EQUAL_DOUBLE(4.0, C->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(3.0, C->vals[1]);

    tensor_free_recursive(C);
}

void test_backward_el_add(void) {
    tg_tensor_t* A = NULL;
    tg_tensor_t* B = NULL;
    TENSOR_CREATE_FILLED(&A, 2.0, 2);
    TENSOR_CREATE_FILLED(&B, 3.0, 2);
    A->vals[1] = 5.0;
    B->vals[1] = 7.0;

    tg_tensor_t* C = tensor_el_add(A, B);

    TEST_ASSERT_EQUAL_DOUBLE(5.0, C->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(12.0, C->vals[1]);

    TENSOR_GRADS_SET(C, 1.0);
    C->backward(C);

    TEST_ASSERT_EQUAL_DOUBLE(1.0, A->grads[0]);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, A->grads[1]);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, B->grads[0]);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, B->grads[1]);

    tensor_free_recursive(C);
}

void test_backward_el_sub(void) {
    tg_tensor_t* A = NULL;
    tg_tensor_t* B = NULL;
    TENSOR_CREATE_FILLED(&A, 8.0, 2);
    TENSOR_CREATE_FILLED(&B, 3.0, 2);
    A->vals[1] = 10.0;
    B->vals[1] = 4.0;

    tg_tensor_t* C = tensor_el_sub(A, B);

    TEST_ASSERT_EQUAL_DOUBLE(5.0, C->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(6.0, C->vals[1]);

    TENSOR_GRADS_SET(C, 1.0);
    C->backward(C);

    TEST_ASSERT_EQUAL_DOUBLE(1.0, A->grads[0]);
    TEST_ASSERT_EQUAL_DOUBLE(1.0, A->grads[1]);
    TEST_ASSERT_EQUAL_DOUBLE(-1.0, B->grads[0]);
    TEST_ASSERT_EQUAL_DOUBLE(-1.0, B->grads[1]);

    tensor_free_recursive(C);
}

void test_backward_el_mul(void) {
    tg_tensor_t* A = NULL;
    tg_tensor_t* B = NULL;
    TENSOR_CREATE_FILLED(&A, 2.0, 2);
    TENSOR_CREATE_FILLED(&B, 4.0, 2);
    A->vals[1] = 3.0;
    B->vals[1] = 5.0;

    tg_tensor_t* C = tensor_el_mul(A, B);

    TEST_ASSERT_EQUAL_DOUBLE(8.0, C->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(15.0, C->vals[1]);

    TENSOR_GRADS_SET(C, 1.0);
    C->backward(C);

    TEST_ASSERT_EQUAL_DOUBLE(4.0, A->grads[0]);
    TEST_ASSERT_EQUAL_DOUBLE(5.0, A->grads[1]);
    TEST_ASSERT_EQUAL_DOUBLE(2.0, B->grads[0]);
    TEST_ASSERT_EQUAL_DOUBLE(3.0, B->grads[1]);

    tensor_free_recursive(C);
}

void test_backward_el_div(void) {
    tg_tensor_t* A = NULL;
    tg_tensor_t* B = NULL;
    TENSOR_CREATE_FILLED(&A, 8.0, 2);
    TENSOR_CREATE_FILLED(&B, 2.0, 2);
    A->vals[1] = 12.0;
    B->vals[1] = 4.0;

    tg_tensor_t* C = tensor_el_div(A, B);

    TEST_ASSERT_EQUAL_DOUBLE(4.0, C->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(3.0, C->vals[1]);

    TENSOR_GRADS_SET(C, 1.0);
    C->backward(C);

    TEST_ASSERT_EQUAL_DOUBLE(0.5, A->grads[0]);
    TEST_ASSERT_EQUAL_DOUBLE(0.25, A->grads[1]);
    TEST_ASSERT_EQUAL_DOUBLE(-2.0, B->grads[0]);
    TEST_ASSERT_EQUAL_FLOAT(-0.75, B->grads[1]);

    tensor_free_recursive(C);
}

void test_backward_gradient_accumulation(void) {
    tg_tensor_t* A = NULL;
    tg_tensor_t* X = NULL;
    tg_tensor_t* Y = NULL;
    TENSOR_CREATE_FILLED(&A, 2.0, 1);
    TENSOR_CREATE_FILLED(&X, 3.0, 1);
    TENSOR_CREATE_FILLED(&Y, 4.0, 1);

    tg_tensor_t* B = tensor_el_mul(A, X);
    tg_tensor_t* C = tensor_el_mul(A, Y);
    tg_tensor_t* D = tensor_el_add(B, C);

    TEST_ASSERT_EQUAL_DOUBLE(6.0, B->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(8.0, C->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(14.0, D->vals[0]);

    TENSOR_GRADS_SET(D, 1.0);
    D->backward(D);

    TEST_ASSERT_EQUAL_DOUBLE(7.0, A->grads[0]);
    TEST_ASSERT_EQUAL_DOUBLE(2.0, X->grads[0]);
    TEST_ASSERT_EQUAL_DOUBLE(2.0, Y->grads[0]);

    tensor_free_recursive(D);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_tensor_init_creates_tensor);
    RUN_TEST(test_tensor_init_multiple_dims);
    RUN_TEST(test_tensor_scalar_add);
    RUN_TEST(test_tensor_scalar_add_negative);
    RUN_TEST(test_tensor_sqrt_all);
    RUN_TEST(test_tensor_shape_init_creates_shape);
    RUN_TEST(test_tensor_shape_total_elements_calculation);
    RUN_TEST(test_tensor_dot_product_simple_calculation);
    RUN_TEST(test_backward_simple_multiplication);
    RUN_TEST(test_backward_gradient_accumulation);
    RUN_TEST(test_backward_leaf_tensor_no_op);
    RUN_TEST(test_reference_counting_prevents_double_free);
    RUN_TEST(test_tensor_free_recursive_complex_dag);
    RUN_TEST(test_forward_el_add);
    RUN_TEST(test_forward_el_sub);
    RUN_TEST(test_forward_el_div);
    RUN_TEST(test_backward_el_add);
    RUN_TEST(test_backward_el_sub);
    RUN_TEST(test_backward_el_mul);
    RUN_TEST(test_backward_el_div);

    return UNITY_END();
}
