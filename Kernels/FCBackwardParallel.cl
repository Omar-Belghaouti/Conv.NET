__kernel void 
FCBackwardParallel(	__global write_only float * deltaXbatch,	// arg 0
					__global read_only float * deltaYbatch, 	// arg 1
					__global read_only float * weights, 		// arg 2
					const int nInputUnits, 						// arg 3
					const int nOutputUnits, 					// arg 4
					const int miniBatchSize						// arg 5
				)
{

	const int iInputUnit = get_global_id(0);
	const int iMiniBatchItem = get_global_id(1);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iInputUnit < nInputUnits && iMiniBatchItem < miniBatchSize)
	{
		int iMiniBatchStart = iMiniBatchItem*nOutputUnits;
		
		float sum = 0.0;
		
		for(int iOutputUnit = 0; iOutputUnit < nOutputUnits; iOutputUnit++)
		{
			// Get element of W^T
			float weightElement = weights[iOutputUnit * nInputUnits + iInputUnit];
			
			// Get element of output gradient deltaY
			float deltaYElement = deltaYbatch[iMiniBatchStart + iOutputUnit];

			// Multiply & cumulate in sum
			sum += weightElement * deltaYElement;
		}
		
		deltaXbatch[iMiniBatchItem*nInputUnits + iInputUnit] = sum;
	}

}