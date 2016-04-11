__kernel void 
ReLUBackward(	__global float * deltaX, 	// arg 0
				__global float * deltaY, 	// arg 1
				__global float * x,			// arg 2
				const int nUnits) 			// arg 3
{
	int iUnit = get_global_id(0);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output array, 
	// therefore it is important to check that global indexes are within the array. The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iUnit < nUnits) 
	{
		if (x[iUnit] < 0)
			deltaX[iUnit] = 0.0;
		else
			deltaX[iUnit] = deltaY[iUnit];
	}
}