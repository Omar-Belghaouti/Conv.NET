__kernel void 
TanhBackward(	__global float * deltaX, 
				__global float * deltaY,
				__global float * y,
				const float beta,
				const int nUnits) 
{
	int iUnit = get_global_id(0);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output array, 
	// therefore it is important to check that global indexes are within the array. The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iUnit < nUnits)
	{
		deltaX[iUnit] = deltaY[iUnit] * (1 - pown(y[iUnit], 2) );
	}
}