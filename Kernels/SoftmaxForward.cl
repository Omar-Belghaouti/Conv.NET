// can be improved _A LOT_!!

__kernel void 
SoftmaxForward (__global float* y,
				__global float* x,
				__global float* auxFloat,
				const int nUnits
                )
{
	uint global_id = get_global_id(0);	
	
	if (global_id == 0) {
		float maxInput = -INFINITY;
		for (uint i = 0; i < nUnits; i++)
		{
			if (x[i] > maxInput)
				maxInput = x[i];
		}
		*auxFloat = maxInput;
		
		
	}

	barrier(CLK_GLOBAL_MEM_FENCE);
	
	y[global_id] = exp(x[global_id] - *auxFloat);
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	if (global_id == 0) {
		float sum = 0.0;
		for (uint i = 0; i < nUnits; i++) {
			sum += y[i];
		}
		*auxFloat = sum;
	}
	
	barrier(CLK_GLOBAL_MEM_FENCE);
	
	y[global_id] /= (*auxFloat);
}