__kernel void 
FCUpdateParameters(	__global float * w,		// arg 0
					__global float * b, 		// arg 1
					__global float * wSpeed, 	// arg 2
					__global float * bSpeed, 	// arg 3
					const int nInput,					// arg 4
					const int nOutput,					// arg 5
					const float weightDecayCoeff
					)
					
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(i < nOutput && j < nInput)
	{
		int iWeight = i*nInput + j;
		w[iWeight] += wSpeed[iWeight] - weightDecayCoeff * w[iWeight];
		
		if (j == 0) // this should be done once per output unit, NOT nInput times!
		{
			b[i] += bSpeed[i];
		}
	}
}