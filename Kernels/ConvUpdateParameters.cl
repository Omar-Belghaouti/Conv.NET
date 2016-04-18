__kernel void 
ConvUpdateParameters(	__global float * w,				// arg 0
						__global float * b, 			// arg 1
						__global float * wSpeed, 		// arg 2
						__global float * bSpeed, 		// arg 3
						const int nFilters,				// arg 4
						const int receptiveFieldSize	// arg 5
					)
					
{
	const int iFilter = get_global_id(0); // index of output row (corresponds to one filter)
	const int iElement = get_global_id(1); // index of output col (corresponds to an element of receptive field iFilter)
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iFilter < nFilters && iElement < receptiveFieldSize)
	{
		const int iWeight = iFilter * receptiveFieldSize + iElement;
		const int iBias = iFilter;
		
		w[iWeight] += wSpeed[iWeight];
		
		if (iElement == 0) // no need to do this multiple times
		{
			b[iBias] += bSpeed[iBias];
		}
	}
}