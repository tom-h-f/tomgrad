#include "unity.h"
#include "../tomgrad/tomgrad.h"
#include <math.h>
#include <stdlib.h>

void setUp(void) {}
void tearDown(void) {}

void test_tensor_init_creates_tensor(void) {
    int dims[] = {3};
    tg_tensor_t* tensor = (tg_tensor_t*)NULL;
    tg_err_t err = tensor_init(dims, 1, tensor);
    TEST_ASSERT_EQUAL(SUCCESS, err);
    TEST_ASSERT_EQUAL(3, tensor->size);
    TEST_ASSERT_NOT_NULL(tensor->vals);
    free(tensor->vals);
}

void test_tensor_init_multiple_dims(void) {
    int dims[] = {2, 3};
    auto tensor = (tg_tensor_t*)NULL;
    tg_err_t err = tensor_init(dims, 1, tensor);
    TEST_ASSERT_EQUAL(SUCCESS, err);
    TEST_ASSERT_EQUAL(5, tensor->size);
    TEST_ASSERT_NOT_NULL(tensor->vals);
    free(tensor->vals);
}

void test_tensor_init_zero_dims(void) {
		int dims [] = {0};
    auto tensor = (tg_tensor_t*)NULL;
    tg_err_t err = tensor_init(dims, 1, tensor);
    TEST_ASSERT_EQUAL(SUCCESS, err);
    TEST_ASSERT_EQUAL(0, tensor->size);
    free(tensor->vals);
}

void test_tensor_scalar_add(void) {
    int dims[] = {3};
    auto tensor = (tg_tensor_t*)NULL;
    tg_err_t err = tensor_init(dims, 1, tensor);
    tensor_init(dims, 1, tensor);
    tensor->vals[0] = 1.0;
    tensor->vals[1] = 2.0;
    tensor->vals[2] = 3.0;
    
    tensor_scalar_add(tensor, 5.0);
    
    TEST_ASSERT_EQUAL_DOUBLE(6.0, tensor->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(7.0, tensor->vals[1]);
    TEST_ASSERT_EQUAL_DOUBLE(8.0, tensor->vals[2]);
    free(tensor->vals);
}

void test_tensor_scalar_add_negative(void) {
    int dims[] = {2};
    auto tensor = (tg_tensor_t*)NULL;
    tg_err_t err = tensor_init(dims, 1, tensor);
    tensor->vals[0] = 10.0;
    tensor->vals[1] = 5.0;
    
    tensor_scalar_add(&tensor, -3.0);
    
    TEST_ASSERT_EQUAL_DOUBLE(7.0, tensor->vals[0]);
    TEST_ASSERT_EQUAL_DOUBLE(2.0, tensor->vals[1]);
    free(tensor->vals);
}

int main(void) {
    UNITY_BEGIN();
    RUN_TEST(test_tensor_init_creates_tensor);
    RUN_TEST(test_tensor_init_multiple_dims);
    RUN_TEST(test_tensor_init_zero_dims);
    RUN_TEST(test_tensor_scalar_add);
    RUN_TEST(test_tensor_scalar_add_negative);
    return UNITY_END();
}
