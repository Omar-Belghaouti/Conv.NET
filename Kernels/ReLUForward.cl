__kernel void 
ReLUForward(__global float * activations, // arg 0
			__global float * preActivations, // arg 1
			int nUnits) 		// arg 2
{
	int iUnit = get_global_id(0);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output array, 
	// therefore it is important to check that global indexes are within the array. The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iUnit < nUnits)
	{
		if (preActivations[iUnit] < 0)
			activations[iUnit] = 0.0;
		else
			activations[iUnit] = preActivations[iUnit];
	}
}