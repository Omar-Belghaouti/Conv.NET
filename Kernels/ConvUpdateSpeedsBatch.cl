__kernel void 
ConvUpdateSpeedsBatch(	__global float * wSpeeds,
						__global float * bSpeeds,
						__global float * deltaOutputBatch,
						__global float * inputBatch,
						__global int * lookupTable,
						const int nFilters, 		
						const int receptiveFieldSize,
						const int nReceptiveFields,
						const float momCoeff,
						const float learningRate,
						const int miniBatchSize
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
		
		const int iDeltaOutputRow = iFilter * nReceptiveFields;
		const int iInputColumn = iElement * nReceptiveFields;
		
		float gradientWeight = 0.0;
		float gradientBias = 0.0;
		
		// Compute gradients, remembering that:
		//    -	WEIGHTS gradients are obtained by multiplying the matrix of error signals deltaOutput 
		//		and the transpose of the input receptive field matrix (which we will create on-the-fly
		// 		by accessing input array using the lookupTable create in the beginning). 
		//    -	BIASES gradients are obtained by simply summing all columns of the matrix of 
		//		error signals deltaOutput, thus obtaining a vector of the same size as the bias vector.
		// All these gradients must be computed for all examples in the mini-batch, and the resulting 
		// values should be averaged and then used to update speeds.
		
		for(int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			const int iExampleBeginningInDeltaOutput = iExample * (nFilters * nReceptiveFields);
			const int iExampleBeginningInInput = iExample * (receptiveFieldSize * nReceptiveFields);
			
			for(int iReceptiveField = 0; iReceptiveField < nReceptiveFields; iReceptiveField++)
			{
				// Get error signal corresponding to this filter and this receptiveField in this example
				int iDeltaOutput = iExampleBeginningInDeltaOutput + iDeltaOutputRow + iReceptiveField;
				float deltaElement = deltaOutputBatch[iDeltaOutput];
				
				// Get input value needed, reading it from transpose(input) indexed using the receptive field lookup table
				int iInput = iExampleBeginningInInput + lookupTable[iInputColumn + iReceptiveField];
				float inputElement = inputBatch[iInput];
				
				// Multiply & cumulate in gradientWeight
				gradientWeight += deltaElement * inputElement;
				
				// Once per filter, cumulate error signals in gradientBias
				if (iElement == 0)
				{
					gradientBias += deltaElement;
				}
			}
		}
		// Now we have the cumulated gradients: need to divide them by miniBatchSize;
		gradientWeight /= miniBatchSize;
		gradientBias /= miniBatchSize;
		
		// Now update weight speed
		wSpeeds[iWeight] = (momCoeff * wSpeeds[iWeight]) - learningRate * gradientWeight;
		
		// And bias speed (again, only do this once per filter)
		if (iElement == 0)
		{
			bSpeeds[iBias] = (momCoeff * bSpeeds[iBias]) - learningRate * gradientBias;
		}
	}
}