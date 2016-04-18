__kernel void 
ConvUpdateSpeeds(	__global float * wSpeeds,
					__global float * bSpeeds,
					__global float * deltaY,
					__global float * input,
					__global int * lookupTable,
					const int nFilters, 		
					const int receptiveFieldSize,
					const int nReceptiveFields,
					const float momCoeff,
					const float learnRatePerItem,
					const int miniBatchItem
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
			
		float tmpSpeedW = wSpeeds[iWeight];
		float tmpSpeedB = bSpeeds[iBias];
		
		// 1. Speed decay (only occurs once when looping through the mini-batch, e.g. if miniBatchItem == 0)
		
		if (miniBatchItem == 0)
		{
			tmpSpeedW *= momCoeff;
			tmpSpeedB *= momCoeff;
		}
		
		float gradW = 0.0;
		float gradB = 0.0;
		
		// 2. And now update speeds based on gradients (this instead occurs at each mini-batch item).
		// Remember that:
		//    -	the WEIGHTS gradients are obtained by multiplying the matrix of error signals deltaY 
		//		and the transpose of the input receptive field matrix (which we will create on-the-fly
		// 		by accessing input array using the lookupTable create in the beginning).
		//    -	the BIASES gradients are obtained by simply summing all columns of the matrix of 
		//		error signals deltaY, thus obtaining a vector of the same size as the bias vector.
		
		for(int iReceptiveField = 0; iReceptiveField < nReceptiveFields; iReceptiveField++)
		{
			// Get error signal corresponding to this filter and this receptiveField
			float deltaElement = deltaY[iFilter * nReceptiveFields + iReceptiveField];
			
			// Get input value needed, reading it from transpose(input) indexed using the receptive field lookup table
			float inputElement = input[ lookupTable[iElement * nReceptiveFields + iReceptiveField] ];
			
			// Multiply & cumulate in gradW
			gradW += deltaElement * inputElement;
			
			// Once per filter, cumulate error signals in gradB
			if (iElement == 0)
			{
				gradB += deltaElement;
			}
		}
		
		// Update weight speed
		tmpSpeedW -= learnRatePerItem * gradW;
		wSpeeds[iWeight] = tmpSpeedW;
		
		// Update bias speed
		if (iElement == 0)
		{
			tmpSpeedB -= learnRatePerItem * gradB;
			bSpeeds[iBias] = tmpSpeedB;
		}
	}
}

