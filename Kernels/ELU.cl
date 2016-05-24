
__kernel void 
ELUForward(__global float * activationsBatch,		// arg 0
			__global float * preActivationsBatch,	// arg 1
			const float alpha,						// arg 2
			const int nTotActivations				// arg 3
			) 						
{
	int i = get_global_id(0);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output array, 
	// therefore it is important to check that global indexes are within the array, The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD),
	
	if(i < nTotActivations) // nTotActivations equals nInputUnits * miniBatchSize
	{
		if (preActivationsBatch[i] < 0)
			activationsBatch[i] = alpha * (exp(preActivationsBatch[i]) - 1.0f);
		else
			activationsBatch[i] = preActivationsBatch[i];
	}
}



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
	// therefore it is important to check that global indexes are within the array, The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD),
	
	if(i < nTotActivations) 
	{
		if (inputBatch[i] < 0.0F)
		{
			float derivative = alpha * exp(inputBatch[i]);
			deltaXbatch[i] = derivative * deltaYbatch[i];
		}
		else
			deltaXbatch[i] = deltaYbatch[i];
	}
}