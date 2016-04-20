__kernel void 
ConvForward(	__global float * output,
				__global float * input,
				__global int * lookupTable, 
				__global float * weights,
				__global float * biases,
				const int nFilters, 		
				const int receptiveFieldSize,
				const int nReceptiveFields
			)
{

	const int iFilter = get_global_id(0); // index of output row (corresponds to one filter)
	const int iReceptiveField = get_global_id(1); // index of output col (corresponds to a receptive field)
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iFilter < nFilters && iReceptiveField < nReceptiveFields)
	{
		float sum = 0.0;
		
		for(int iElement = 0; iElement < receptiveFieldSize; iElement++)
		{
			// Get filter element needed 
			float filterElement = weights[iFilter * receptiveFieldSize + iElement];
			
			// Get receptive field element needed, reading it from 
			// inputPadded indexed using the receptive field lookup table
			float receptiveFieldElement = input[ lookupTable[iElement * nReceptiveFields + iReceptiveField] ];
			
			// Multiply & cumulate in sum
			sum += filterElement * receptiveFieldElement;
		}
		
		// Add bias
		sum += biases[iFilter];
		
		// Finally, write output buffer
		output[iFilter * nReceptiveFields + iReceptiveField] = sum;
	}
}

