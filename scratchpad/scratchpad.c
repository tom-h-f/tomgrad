#include "../tomgrad/tomgrad.h"
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>

int main(void) {
		int dims[] = {6, 6, 6};
		tg_tensor_t* ptr = (tg_tensor_t*)NULL;
		UNWRAP(tensor_init(dims, 2, &ptr));
		tensor_scalar_add(ptr, 2.0);
		return 0;
}
