__kernel void 
ReLUBackward(	__global write_only float * deltaXbatch,	// arg 0
				__global read_only float * deltaYbatch, 	// arg 1
				__global read_only float * inputBatch,		// arg 2
				const int nTotActivations) 					// arg 3
{
	int i = get_global_id(0);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output array, 
	// therefore it is important to check that global indexes are within the array. The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(i < nTotActivations) 
	{
		if (inputBatch[i] <= 0)
			deltaXbatch[i] = 0.0;
		else
			deltaXbatch[i] = deltaYbatch[i];
	}
}