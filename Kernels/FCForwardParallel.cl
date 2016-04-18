__kernel void 
FCForwardParallel(	__global write_only float * outputBatch,	// arg 0
					__global read_only float * inputBatch, 		// arg 1
					__global read_only float * weights, 		// arg 2
					__global read_only float * biases, 			// arg 3
					const int nInputUnits, 						// arg 4
					const int nOutputUnits, 					// arg 5
					const int miniBatchSize						// arg 6
				)
{

	const int iOutputUnit = get_global_id(0);
	const int iMiniBatchItem = get_global_id(1);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iOutputUnit < nOutputUnits && iMiniBatchItem < miniBatchSize)
	{
		int iMiniBatchStart = iMiniBatchItem * nInputUnits;
	
		float sum = 0.0;
		
		for(int iInputUnit = 0; iInputUnit < nInputUnits; iInputUnit++)
		{
			// Get weight element
			float weightElement = weights[iOutputUnit * nInputUnits + iInputUnit];
			
			// Get input element
			float inputElement = inputBatch[iMiniBatchStart + iInputUnit];

			// Multiply & cumulate in sum
			sum += weightElement * inputElement;
		}
		
		// Add bias
		sum += biases[iOutputUnit];
		
		outputBatch[iMiniBatchItem*nOutputUnits + iOutputUnit ] = sum;
	}

}