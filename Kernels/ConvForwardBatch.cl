__kernel void 
ConvForwardBatch(	__write_only __global float * outputBatch,
					__read_only __global float * inputBatch, // already padded (if necessary)
					__read_only __global int * lookupTable, 
					__read_only __global float * weights,
					__read_only __global float * biases,
					const int nFilters, 		
					const int receptiveFieldSize,
					const int nReceptiveFields,
					const int inputVolume,
					const int miniBatchSize
				)
{

	const int iRow = get_global_id(0); // index of output row (corresponds to one filter)
	const int iReceptiveField = get_global_id(1); // index of output col (corresponds to a receptive field)
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iRow < nFilters * miniBatchSize && iReceptiveField < nReceptiveFields)
	{
		const int iMiniBatchItem = iRow / miniBatchSize;
		const int iFilter = iRow % miniBatchSize;
		
		const int iInputMiniBatchItemBeginning = iMiniBatchItem * inputVolume;
		const int iFilterRowBeginning = iFilter * receptiveFieldSize;
		
		float sum = 0.0;
		
		for(int iElement = 0; iElement < receptiveFieldSize; iElement++)
		{
			// Get filter element needed 
			float filterElement = weights[iFilterRowBeginning + iElement];
			
			// Get receptive field element needed, reading it from inputBatch using the lookup table
			int iInput = iInputMiniBatchItemBeginning + lookupTable[iElement * nReceptiveFields + iReceptiveField];
			float receptiveFieldElement = inputBatch[iInput];
			
			// Multiply & cumulate in sum
			sum += filterElement * receptiveFieldElement;
		}
		
		// Add bias
		sum += biases[iFilter];
		
		// Finally, write resulting sum into outputBatch buffer
		const int iOutputMiniBatchItemBeginning = iMiniBatchItem * nFilters * nReceptiveFields;
		const int iOutput = iOutputMiniBatchItemBeginning + iFilter * nReceptiveFields + iReceptiveField;
		outputBatch[iOutput] = sum;
	}
	
}

