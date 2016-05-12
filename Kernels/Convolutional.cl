__kernel void 
CreateRecFieldsLookupTable (__global int* recFieldLookupTable,
							const int inputWidth,  // already takes the padding into account
							const int outputWidth, // already takes the stride into account
							const int filterSize,
							const int receptiveFieldSize,
							const int stride
							)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
	
	const int nReceptiveFields = outputWidth * outputWidth;
	
	if (i < receptiveFieldSize && j < nReceptiveFields) // check if we are inside the matrix
	{
		const int iReceptiveFieldElement = i;
		const int iReceptiveField = j; 
		
		int iInput = 0; // will be incremented as we "zoom in" step by step
		const int iOutput = iReceptiveFieldElement * nReceptiveFields + iReceptiveField; // destination index
		
		// 1. move to the beginning of channel that we are working on (using i)
		const int iChannel = i / (filterSize * filterSize);
		const int elementsPerChannel = inputWidth*inputWidth;
		const int iBeginningOfChannel = elementsPerChannel * iChannel;
		
		iInput += iBeginningOfChannel;
		
		// 2. now move to the beginning of the receptive field that we are working on (using j)
		// (remember that we are already at the beginning of the correct channel!) 
		const int iOutputRow = j / outputWidth;
		const int iOutputCol = j % outputWidth;
		const int iBeginningOfReceptiveField = iOutputRow * stride * inputWidth + stride * iOutputCol;
		
		iInput += iBeginningOfReceptiveField;
		
		// 3. now move to the correct position within the current receptive field (again, using i)
		// (remember that we are already in the correct channel and receptive field!)
		const int iFilterRow = (i % (filterSize * filterSize)) / filterSize;
		const int iReceptiveFieldCol = i % filterSize;
		const int iWithinReceptiveField = inputWidth * iFilterRow + iReceptiveFieldCol;
		
		iInput += iWithinReceptiveField;
		
		recFieldLookupTable[iOutput]= iInput;
	}
	
}


__kernel void 
CreatePaddingLookupTable (	__global int* paddingLookupTable,
							const int unpaddedWidth, // only Height equal to Width is supported
							const int unpaddedDepth,
							const int padding
						)
{
	const int iUnpadded = get_global_id(0); // index of input element within current example
	
	// 0. Compute some useful quantities
		const int unpaddedArea = unpaddedWidth * unpaddedWidth;
		const int unpaddedVolume = unpaddedDepth * unpaddedArea;
		const int nZerosTopRows = padding * (2 * padding + unpaddedWidth);
		const int nZerosPerChannel = 4 * padding * (unpaddedWidth + padding);
		
	if (iUnpadded < unpaddedVolume) // this is important because of how local/global work sizes are set (more efficient)
	{	
		// 1. Initialize iPadded equal to iUnpadded
		int iPadded = iUnpadded;
		
		// 2. Find index of unpadded slice/channel that we are working on...
		int iChannel = (iUnpadded % unpaddedVolume) / unpaddedArea;
		/// ...and add the number of zeros padding all channels before this one
		iPadded += nZerosPerChannel * iChannel;
		
		// 3. Find index of row that we are working on...
		int iRow = (iUnpadded % unpaddedArea) / unpaddedWidth;
		// ...and add the number of zeros padding all rows before this one
		iPadded += nZerosTopRows + padding * (2*iRow + 1);
		
		// 4. Finally, write the resulting index in the lookup table
		paddingLookupTable[iUnpadded] = iPadded;
	}

}


/* 
 * OpenCL kernel for zero-padding an input vector containing a mini-batch
 * Entries of the input arrays are rewritten to a new position, accounting for padding.
 */
 
 __kernel void 
 ZeroPad(	__global float* paddedInputBatch,
			__global float* inputBatch,
			__global int* paddingLookupTable,
			const int unpaddedVolume,
			const int paddedVolume,
			const int miniBatchSize
			)
{	
	const int iUnpadded = get_global_id(0);
	
	if (iUnpadded < unpaddedVolume * miniBatchSize) // this is important because of how local/global work sizes are set (more efficient)
	{

		int iExample = iUnpadded / unpaddedVolume;
		int iUnpaddedWithinExample = iUnpadded - unpaddedVolume*iExample;
		
		// Find beginning of mini-batch item that we are working on in *padded* array
		int iExampleBeginningInPadded = iExample * paddedVolume;
		
		// Find index of destination element (in padded array) using the lookup table
		int iPadded = iExampleBeginningInPadded + paddingLookupTable[iUnpaddedWithinExample];
		
		// Write value
		paddedInputBatch[iPadded] = inputBatch[iUnpadded];
	}
}


/* 
 * OpenCL kernel for UNpadding vector "paddedArrayBatch".
 */
 
 __kernel void 
 ZeroUnpad(__global float* unpaddedArrayBatch,
			__global float* paddedArrayBatch,
			__global int* paddingLookupTable,
			const int unpaddedVolume,
			const int paddedVolume,
			const int miniBatchSize
			)
{	
	const int iUnpadded = get_global_id(0);
		
	if (iUnpadded < unpaddedVolume * miniBatchSize) // this is important because of how local/global work sizes are set (more efficient)
	{
		int iExample = iUnpadded / unpaddedVolume;
		int iUnpaddedWithinExample = iUnpadded - unpaddedVolume*iExample;
		
		// Find beginning of mini-batch item that we are working on in *padded* array
		int iExampleBeginningInPadded = iExample * paddedVolume;
		
		// Find index of source element (in padded array) using the lookup table
		int iPadded = iExampleBeginningInPadded + paddingLookupTable[iUnpaddedWithinExample];
		
		// Write value
		unpaddedArrayBatch[iUnpadded] = paddedArrayBatch[iPadded];
	}
}



/* 
 * OpenCL kernel for forward pass of ConvolutionalLayer class,
 * implemented as a matrix multiplication between a filter matrix and a matrix of input receptive fields,
 * constructed on-the-fly using a pre-constructed "lookup table". Then biases are added. 
 * Input/output arrays actually contain a mini-batch of i/o examples.
 */

__kernel void 
ConvForward(__global float * outputBatch,
			__global float * inputBatch, // already padded (if necessary)
			__global int * lookupTable, 
			__global float * weights,
			__global float * biases,
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
		const int iMiniBatchItem = iRow / nFilters;
		const int iFilter = iRow - nFilters * (iMiniBatchItem);
		
		const int iOutputMiniBatchItemBeginning = iMiniBatchItem * nFilters * nReceptiveFields;
		const int iOutput = iOutputMiniBatchItemBeginning + iFilter * nReceptiveFields + iReceptiveField;
		
		const int iInputMiniBatchItemBeginning = iMiniBatchItem * inputVolume;
		const int iFilterRowBeginning = iFilter * receptiveFieldSize;
		
		float sum = 0.0;
		int iInput = 0;
		
		for(int iElement = 0; iElement < receptiveFieldSize; ++iElement)
		{
			// Get receptive field element needed, reading it from inputBatch using the lookup table
			iInput = iInputMiniBatchItemBeginning + lookupTable[iElement * nReceptiveFields + iReceptiveField];
			
			// Multiply & cumulate in sum
			sum = fma(weights[iFilterRowBeginning + iElement], inputBatch[iInput], sum);
		}
		
		// Add bias
		sum += biases[iFilter];
		
		// Finally, write resulting sum into outputBatch buffer
		outputBatch[iOutput] = sum;
		
	}
	
}


/* 
 * OpenCL kernel for gradient backpropagation in convolutional layers (deltaY to deltaX)
 * implemented as a matrix multiplication between transpose(weights) and deltaY. Results 
 * are written directly into deltaX (and not into a deltaReceptiveFields matrix), using
 * the pre-computed lookup table. For this reason, it is IMPORTANT to remember to wipe off
 * deltaX (write zeros) before calling this kernel. 
 * All of this is done in parallel across a mini-batch of output gradients.
 */

__kernel void 
ConvBackPropagate(	__global float * deltaInputBatch,
					const int inputVolume,				// this already includes padding, if any!!
					__global float * deltaOutputBatch,
					__global float * weights,
					__global int * recFieldslookupTable,
					const int nFilters, 		
					const int receptiveFieldSize,
					const int nReceptiveFields,
					const int miniBatchSize
					)
{

	const int iRow = get_global_id(0); // index of output row
	const int iReceptiveField = get_global_id(1); // index of output col (corresponds to a receptive field)
	
	/*
     *	Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	 *	therefore it is important to check that global indexes are within the matrix. The computational cost 
	 *	of these comparisons is greatly compensated by the increased efficiency of using a local work size
	 * 	that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	 */
	 
	
	if(iRow < receptiveFieldSize * miniBatchSize && iReceptiveField < nReceptiveFields)
	{
		const int iExample = iRow / receptiveFieldSize;
		const int iRecFieldElement = iRow - receptiveFieldSize * iExample; // equivalent to iRow % receptiveFieldSize
		
		const int iExampleBeginningInDeltaOutput = iExample * (nFilters * nReceptiveFields);
		
		float tmpDeltaInput = 0.0F;
		
		int iDeltaOutput = 0;
		
		for(int iFilter = 0; iFilter < nFilters; ++iFilter)
		{
			// Get error signal corresponding to this mini-batch item, this filter, and this receptiveField:
			// first move to the beginning of this example, then pick the right "row and column"
			iDeltaOutput = iExampleBeginningInDeltaOutput + iFilter * nReceptiveFields + iReceptiveField;
			
			// Multiply & cumulate in tmpDeltaInput
			tmpDeltaInput += weights[iFilter * receptiveFieldSize + iRecFieldElement] * deltaOutputBatch[iDeltaOutput];
		}
		
		// Now cumulate this portion of gradient into the correct position of paddedDeltaX (using lookup table)
		// This way, error signals coming from different receptive field positions (but corresponding to the
		// same input position) will be added up, as it should be. (Can be proven on paper.)
		const int iExampleBeginningInDeltaInput = iExample * inputVolume;
		const int iFromLookupTable = recFieldslookupTable[iRecFieldElement * nReceptiveFields + iReceptiveField];
		const int iDeltaInput = iExampleBeginningInDeltaInput + iFromLookupTable;
		deltaInputBatch[iDeltaInput] += tmpDeltaInput;
		
	}
	
}


/* 
 * OpenCL kernel for updating weights/biases speed in Convolutional layers
 * using the gradient computed with backpropagation, for a mini-batch
 * of inputs / delta signals.
 */

__kernel void ConvUpdateSpeeds(	__global float * wSpeeds,
								__global float * bSpeeds,
								__global float * wGrad,
								__global float * bGrad,
								__global float * deltaOutputBatch,
								__global float * inputBatch,		// either padded or unpadded! Set argument accordingly
								const int inputVolume, 				// either padded or unpadded! Set argument accordingly
								__global int * recFieldsLookupTable,
								const int nFilters, 		
								const int receptiveFieldSize,
								const int nReceptiveFields,
								const float momCoeff,
								const float learningRate,
								const int miniBatchSize
								)
{

	const int iFilter = get_global_id(0); // index of output row (corresponds to one filter)
	const int iElement = get_global_id(1); // index of output col (corresponds to an element of receptive field of filter iFilter)
	
	/*
	 * Because of how the work group sizes are set, the global work size can be larger than the output matrix, 
	 * therefore it is important to check that global indexes are within the matrix. The computational cost 
	 * of these comparisons is greatly compensated by the increased efficiency of using a local work size
	 * that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	 */
	 
	if(iFilter < nFilters && iElement < receptiveFieldSize)
	{
		const int iWeight = iFilter * receptiveFieldSize + iElement;
		const int iBias = iFilter;
		
		const int iDeltaOutputRow = iFilter * nReceptiveFields;
		const int iInputTransposeColumn = iElement * nReceptiveFields; // equivalent to iInputRow
		
		float gradientWeight = 0.0;
		float gradientBias = 0.0;
		
		/*
		 * Compute gradients, remembering that:
		 *
		 *    -	WEIGHTS gradients are obtained by multiplying the matrix of error signals deltaOutput 
		 *		with the transpose of the input receptive field matrix (which we will create on-the-fly
		 * 		by accessing input array using the lookupTable create in the beginning). 
		 *
		 *    -	BIASES gradients are obtained by simply summing all columns of the matrix of 
		 *		error signals deltaOutput, thus obtaining a 1D vector of length nFilters.
		 *
		 * All these gradients must be computed for all examples in the mini-batch, and the resulting 
		 * values should be averaged and then used to update speeds.
		 */
		 
		int iExampleBeginningInInput = 0;
		int iBeginningDeltaOutputRow = 0;
		
		float deltaElement = 0.0F;
		
		int iInput = 0;
		
		const int outputVolume = nFilters * nReceptiveFields;
		
		for(int iExample = 0; iExample < miniBatchSize; ++iExample)
		{
			iExampleBeginningInInput = iExample * inputVolume;
			
			iBeginningDeltaOutputRow = iExample * outputVolume + iDeltaOutputRow;
			
			for(int iReceptiveField = 0; iReceptiveField < nReceptiveFields; ++iReceptiveField)
			{
				// Get error signal corresponding to this filter and this receptiveField in this example:
				// first move to the correct example, then pick the right "row and column"
				deltaElement = deltaOutputBatch[iBeginningDeltaOutputRow + iReceptiveField];
				
				// Get input value needed, reading it from transpose(input): first move to the correct example, 
				// then pick the right "row and column" using the pre-constructed receptive field lookup table
				iInput = iExampleBeginningInInput + recFieldsLookupTable[iInputTransposeColumn + iReceptiveField];
				
				// Multiply & cumulate in gradientWeight
				//gradientWeight += deltaElement * inputBatch[iInput];
				gradientWeight = fma(deltaElement, inputBatch[iInput], gradientWeight);
				
				// Once per filter, also cumulate error signals in gradientBias
				if (iElement == 0)
				{
					gradientBias += deltaElement;
				}
			}
		}
		// Now we have the cumulated gradients: need to divide them by miniBatchSize;
		gradientWeight /= miniBatchSize;
		gradientBias /= miniBatchSize;
		
		// Save gradient
		wGrad[iWeight] = gradientWeight;
		// Update weight speed
		wSpeeds[iWeight] = (momCoeff * wSpeeds[iWeight]) - learningRate * gradientWeight;
		
		// And bias gradient/speed (again, only do this once per filter)
		if (iElement == 0)
		{
			bGrad[iBias] = gradientBias; // save gradient
			bSpeeds[iBias] = (momCoeff * bSpeeds[iBias]) - learningRate * gradientBias;
		}
		
	}
}


__kernel void 
ConvUpdateParameters(	__global float * w,				// arg 0
						__global float * b, 			// arg 1
						__global float * wSpeed, 		// arg 2
						__global float * bSpeed, 		// arg 3
						const int nFilters,				// arg 4
						const int receptiveFieldSize,	// arg 5
						const float weightDecayCoeff	// arg 6
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
		
		w[iWeight] += wSpeed[iWeight] - weightDecayCoeff * w[iWeight];
		
		if (iElement == 0) // this must be done only once per filter
		{
			b[iBias] += bSpeed[iBias];
		}
	}
}


