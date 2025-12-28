#include "unity.h"
#include "../tomgrad/tomgrad.h"
#include "unity_internals.h"

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
    TENSOR_CREATE_RANDOM(&a, 2.0, 2);
    TENSOR_CREATE_FILLED(&b, 2.0, 2);

    tg_value_t result = tensor_dot_product(a, b);
    TEST_ASSERT_EQUAL(result, 8.0);

    tensor_free(a);
    tensor_free(b);
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
    
    return UNITY_END();
}
