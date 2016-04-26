#define EPSILON 0.0000001

__kernel void 
ELUBackward(	__global float * deltaXbatch,	// arg 0
				__global float * deltaYbatch, 	// arg 1
				__global float * inputBatch,	// arg 2
				const float alpha,				// arg 3
				const int nTotActivations		// arg 4
				) 					
{
	int i = get_global_id(0);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output array, 
	// therefore it is important to check that global indexes are within the array. The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(i < nTotActivations) 
	{
		if (inputBatch[i] < - EPSILON)
		{
			float derivative = alpha * exp(inputBatch[i]);
			deltaXbatch[i] = derivative * deltaYbatch[i];
		}
		else if (inputBatch[i] > EPSILON)	
			deltaXbatch[i] = deltaYbatch[i];
		else
			deltaXbatch[i] = 0.0f; // trick: when -EPS < input < EPS we assume it has been dropped out => no backprop
	}
}