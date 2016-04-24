__kernel void 
ReLUForward(__global write_only float * activationsBatch,	// arg 0
			__global read_only float * preActivationsBatch,	// arg 1
			const int nTotActivations						// arg 2
			) 						
{
	int i = get_global_id(0);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output array, 
	// therefore it is important to check that global indexes are within the array. The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(i < nTotActivations) // nTotActivations equals nInputUnits * miniBatchSize
	{
		if (preActivationsBatch[i] <= 0)
			activationsBatch[i] = 0.0;
		else
			activationsBatch[i] = preActivationsBatch[i];
	}
}