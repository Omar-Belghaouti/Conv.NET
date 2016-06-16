/********************************************************************************************
* NOTE: 
* In comments, the word OUTPUT denotes that the kernel argument is actually an output argument
* for the kernel. Attention: this does not necessarily mean that the argument is an output
* object for the layer!
*
**********************************************************************************************/




/*
 * CreateRecFieldsLookupTable()
 * Creates a lookup table for accessing the array of input activations "inputNeurons.ActivationsGPU"
 * AS IF it was reshaped into a set of matrix of unrolled receptive fields.
 * This table is needed in kernels ConvForward, ConvBackPropagate, ConvUpdateSpeeds.
 * To be called ONCE, when the layer is initialised.
 */
__kernel void 
CreateRecFieldsLookupTable (__global int* recFieldLookupTable, // OUTPUT memory buffer. Size (in bytes): sizeof(int) * outputHeight * outputWidth * inputDepth * filterSize^2
							const int inputWidth,  // width of input tensor, INCLUDING padding (i.e. originalInputWidth + 2 * zeroPadding) 
							const int outputWidth, // width of output tensor, i.e. (inputWidth - filterSize + 2 * zeroPadding) / (strideLength + 1) <--- make sure this is integer!
							const int filterSize,
							const int receptiveFieldSize, // inputDepth * filterSize^2
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
		
		// 1, move to the beginning of channel that we are working on (using i)
		const int iChannel = i / (filterSize * filterSize);
		const int elementsPerChannel = inputWidth*inputWidth;
		const int iBeginningOfChannel = elementsPerChannel * iChannel;
		
		iInput += iBeginningOfChannel;
		
		// 2, now move to the beginning of the receptive field that we are working on (using j)
		// (remember that we are already at the beginning of the correct channel!) 
		const int iOutputRow = j / outputWidth;
		const int iOutputCol = j % outputWidth;
		const int iBeginningOfReceptiveField = iOutputRow * stride * inputWidth + stride * iOutputCol;
		
		iInput += iBeginningOfReceptiveField;
		
		// 3, now move to the correct position within the current receptive field (again, using i)
		// (remember that we are already in the correct channel and receptive field!)
		const int iFilterRow = (i % (filterSize * filterSize)) / filterSize;
		const int iReceptiveFieldCol = i % filterSize;
		const int iWithinReceptiveField = inputWidth * iFilterRow + iReceptiveFieldCol;
		
		iInput += iWithinReceptiveField;
		
		recFieldLookupTable[iOutput]= iInput;
	}
	
}


/*
 * CreatePaddingLookupTable()
 * Creates a lookup table for padding the input tensor with a frame of zeros.
 * This table is needed in kernels ZeroPad and ZeroUnpad.
 * To be called ONCE, when the layer is initialised.
 */
__kernel void 
CreatePaddingLookupTable (	__global int* paddingLookupTable, // OUTPUT buffer. Size (in bytes): sizeof(int) * inputDepth * inputHeight * inputWidth 
							const int unpaddedWidth, // inputWidth (only Height equal to Width is supported)
							const int unpaddedDepth, // inputDepth
							const int padding
						)
{
	const int iUnpadded = get_global_id(0); // index of input element within current example
	
	// 0, Compute some useful quantities
		const int unpaddedArea = unpaddedWidth * unpaddedWidth;
		const int unpaddedVolume = unpaddedDepth * unpaddedArea;
		const int nZerosTopRows = padding * (2 * padding + unpaddedWidth);
		const int nZerosPerChannel = 4 * padding * (unpaddedWidth + padding);
		
	if (iUnpadded < unpaddedVolume) // this is important because of how local/global work sizes are set (more efficient)
	{	
		// 1, Initialize iPadded equal to iUnpadded
		int iPadded = iUnpadded;
		
		// 2, Find index of unpadded slice/channel that we are working on...
		int iChannel = (iUnpadded % unpaddedVolume) / unpaddedArea;
		/// ...and add the number of zeros padding all channels before this one
		iPadded += nZerosPerChannel * iChannel;
		
		// 3, Find index of row that we are working on...
		int iRow = (iUnpadded % unpaddedArea) / unpaddedWidth;
		// ...and add the number of zeros padding all rows before this one
		iPadded += nZerosTopRows + padding * (2*iRow + 1);
		
		// 4, Finally, write the resulting index in the lookup table
		paddingLookupTable[iUnpadded] = iPadded;
	}

}


/* 
 * ZeroPad()
 * Pads the input tensor (unrolled in array inputBatch) with zeros, around its spatial dimensions.
 * To be called in the forward pass, before kernel ConvForward.
 */
 
 __kernel void 
 ZeroPad(	__global float* paddedInputBatch, // OUTPUT tensor of activations
			__global float* inputBatch, // Input tensor of activations
			__global int* paddingLookupTable, // Lookup table created with CreatePaddingLookupTable
			const int unpaddedVolume, // inputDepth * inputHeight * inputWidth
			const int paddedVolume, // inputDepth * (inputHeight + 2 * zeroPadding) * (inputWidth + 2 * zeroPadding)
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
 * ZeroUnpad()
 * Un-pads the (padded) tensor of gradients with respect to the input units.
 * To be called in the backward pass, before kernel ConvBackPropagate.
 */
 
 __kernel void 
 ZeroUnpad(__global float* unpaddedArrayBatch, // OUTPUT: array of gradients with respect to the input units, unpadded
			__global float* paddedArrayBatch, // Array of gradients with respect to the input units, padded
			__global int* paddingLookupTable, // Lookup table created with CreatePaddingLookupTable
			const int unpaddedVolume, // inputDepth * inputHeight * inputWidth
			const int paddedVolume, // inputDepth * (inputHeight + 2 * zeroPadding) * (inputWidth + 2 * zeroPadding)
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
 * ConvForward()
 * Forward pass of the ConvolutionalLayer class, implemented as set of matrix multiplications between a filter matrix 
 * and matrices of unrolled input receptive fields constructed on-the-fly using a pre-constructed lookup table.
 */

__kernel void 
ConvForward(__global float * outputBatch, // OUTPUT: tensor of output activations
			__global float * inputBatch, // Tensor of input activation (possibly PADDED)
			__global int * lookupTable, // Lookup table created by CreateRecFieldsLookupTable
			__global float * weights, // weight matrix (GPU buffer)
			__global float * biases, // biases (GPU buffer)
			const int nFilters, 		
			const int receptiveFieldSize, // inputDepth * filterSize * filterSize
			const int nReceptiveFields, // outputHeight * outputWidth
			const int inputVolume, // PADDED input volume, i.e. inputDepth * (inputHeight + 2 * zeroPadding) * (inputWidth + 2 * zeroPadding)
			const int miniBatchSize,
			__global bool * dropoutMask, // OUTPUT dropout mask, created by sampling a pseudo-random number on the GPU (this is bad, but works well)
			const float dropoutParameter, 
			const ulong randomSeed // global seed for the pseudo-random umber generator
				)
{

	const int iRow = get_global_id(0); // index of output row (corresponds to one filter)
	const int iReceptiveField = get_global_id(1); // index of output col (corresponds to a receptive field)
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix, The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD),
	
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
		
		// Dropout here (more efficient: no matrix multiplication if unit is deactivated)
		bool isUnitOn;
		if (dropoutParameter < 1.0F)
		{
			// generate a pseudo-random number here, mimicking Java RNG
			ulong thisSeed = randomSeed + iOutput; // every unit will use a slightly different seed...
			thisSeed = (thisSeed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
			uint pseudoRandomInt = thisSeed >> 16;
			for (int j = 0; j < 6; ++j)
			{
				// ...but this slight difference will then be amplified, by repeating this a few times
				thisSeed = pseudoRandomInt;
				thisSeed = (thisSeed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
				pseudoRandomInt = thisSeed >> 16;
			}
			float pseudoRandFloat = (float)pseudoRandomInt/(float)4294967295;
			// this is not a very good pseudo random number, but hopefully it's good enough
			isUnitOn = pseudoRandFloat < dropoutParameter;
		}
		else
		{
			isUnitOn = true;
		}
		// save unit state
		dropoutMask[iOutput] = isUnitOn;
		
		
		if (isUnitOn)
		{
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
			outputBatch[iOutput] = sum/dropoutParameter;
		}
		else // unit is dropped out
		{
			outputBatch[iOutput] = 0.0f;
		}
	}
}

 // Atomic addition for floats,needed in kernel ConvBackPropagate()
 // NOTE: This is a bit slow, but I didn't found any other solution yet.
inline void AtomicAdd(volatile __global float *addr, float val)
   {
       union{
           unsigned int u32;
           float        f32;
       } next, expected, current;
   	current.f32    = *addr;
       do{
   	   expected.f32 = current.f32;
           next.f32     = expected.f32 + val;
   		current.u32  = atomic_cmpxchg( (volatile __global unsigned int *)addr, 
                               expected.u32, next.u32);
       } while( current.u32 != expected.u32 );
   }
   
   
/* 
 * ConvBackPropagate()
 * Gradient backpropagation in convolutional layers (deltaY to deltaX),
 * implemented as a matrix multiplication between transpose(weights) and deltaY, Results 
 * are written directly into deltaX (and not into a deltaReceptiveFields matrix), using
 * the pre-computed lookup table, For this reason, it is important to wipe deltaX (writing zeros) 
 * in the host code before calling this kernel.
 */
__kernel void 
ConvBackPropagate(	__global float * deltaInputBatch, 	// OUTPUT: tensor of gradients with respect to input units 
					const int inputVolume,				// PADDED input volume, i.e. inputDepth * (inputHeight + 2 * zeroPadding) * (inputWidth + 2 * zeroPadding)
					__global float * deltaOutputBatch,	// Tensor of gradients with respect to output units 
					__global float * weights,			// weight matrix (GPU buffer)
					__global int * recFieldslookupTable,// Lookup table created by CreateRecFieldsLookupTable
					const int nFilters, 				// i.e. number of of feature maps
					const int receptiveFieldSize,		// inputDepth * filterSize * filterSize
					const int nReceptiveFields,			// outputHeight * outputWidth
					const int miniBatchSize,
					__global bool * dropoutMask			// Dropout mask generated in ConvForward()
					)
{

	const int iRow = get_global_id(0); // index of output row
	const int iReceptiveField = get_global_id(1); // index of output col (corresponds to a receptive field)
	
	/*
     *	Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	 *	therefore it is important to check that global indexes are within the matrix, The computational cost 
	 *	of these comparisons is greatly compensated by the increased efficiency of using a local work size
	 * 	that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD),
	 */
	 
	
	if(iRow < receptiveFieldSize * miniBatchSize && iReceptiveField < nReceptiveFields)
	{
		const int iExample = iRow / receptiveFieldSize;
		const int iRecFieldElement = iRow % receptiveFieldSize; // equivalent to iRow % receptiveFieldSize
		
		const int iExampleBeginningInDeltaOutput = iExample * (nFilters * nReceptiveFields);
		
		float tmpDeltaInput = 0.0F;
		
		int iDeltaOutput = 0;
		
		for(int iFilter = 0; iFilter < nFilters; ++iFilter)
		{
			// Get error signal corresponding to this mini-batch item, this filter, and this receptiveField:
			// first move to the beginning of this example, then pick the right "row and column"
			iDeltaOutput = iExampleBeginningInDeltaOutput + iFilter * nReceptiveFields + iReceptiveField;
			
			if (dropoutMask[iDeltaOutput] == true) // if output unit is active
			{
				// Multiply & cumulate in tmpDeltaInput
				tmpDeltaInput += weights[iFilter * receptiveFieldSize + iRecFieldElement] * deltaOutputBatch[iDeltaOutput];
			}
		}
		
		// Now cumulate this portion of gradient into the correct position of paddedDeltaX (using lookup table)
		// This way, error signals coming from different receptive field positions (but corresponding to the
		// same input position) will be added up, as it should be, (Can be proven on paper,)
		const int iExampleBeginningInDeltaInput = iExample * inputVolume;
		const int iFromLookupTable = recFieldslookupTable[iRecFieldElement * nReceptiveFields + iReceptiveField];
		const int iDeltaInput = iExampleBeginningInDeltaInput + iFromLookupTable;
		//deltaInputBatch[iDeltaInput] += tmpDeltaInput;
		AtomicAdd(&deltaInputBatch[iDeltaInput], tmpDeltaInput);
	}
	
}





/* 
 * ConvUpdateSpeeds()
 * Updates weights and biases speed in convolutional layers. These speeds will be then used to update
 * parameters in kernel ConvUpdateSpeeds(), using the momentum update rule
 */

__kernel void ConvUpdateSpeeds(	__global float * wSpeeds, 			// OUTPUT: weights update speeds
								__global float * bSpeeds,			// OUTPUT: biases update speeds
								__global float * wGrad,				// OUTPUT: gradient wrt weights
								__global float * bGrad,				// OUTPUT: gradient wrt biases
								__global float * deltaOutputBatch,	// Tensor of gradients with respect to output units 
								__global float * inputBatch,		// Tensor of input activation (possibly PADDED)
								const int inputVolume, 				// PADDED input volume, i.e. inputDepth * (inputHeight + 2 * zeroPadding) * (inputWidth + 2 * zeroPadding)
								__global int * recFieldsLookupTable,// Lookup table created by CreateRecFieldsLookupTable
								const int nFilters, 		
								const int receptiveFieldSize,		// inputDepth * filterSize * filterSize
								const int nReceptiveFields,			// outputHeight * outputWidth
								const float momCoeff,
								const float learningRate,
								const int miniBatchSize,
								__global bool * dropoutMask,		// Dropout mask generated in ConvForward()
								__global float * weights,			// weight matrix (GPU buffer)
								const float weightDecayCoeff
								)
{

	const int iFilter = get_global_id(0); // index of output row (corresponds to one filter)
	const int iElement = get_global_id(1); // index of output col (corresponds to an element of receptive field of filter iFilter)
	
	/*
	 * Because of how the work group sizes are set, the global work size can be larger than the output matrix, 
	 * therefore it is important to check that global indexes are within the matrix, The computational cost 
	 * of these comparisons is greatly compensated by the increased efficiency of using a local work size
	 * that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD),
	 */
	 
	if(iFilter < nFilters && iElement < receptiveFieldSize)
	{
		const int iWeight = iFilter * receptiveFieldSize + iElement;
		const float thisWeightDecay = weightDecayCoeff*weights[iWeight];
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
		 * 		by accessing input array using the lookupTable create in the beginning), 
		 *
		 *    -	BIASES gradients are obtained by simply summing all columns of the matrix of 
		 *		error signals deltaOutput, thus obtaining a 1D vector of length nFilters,
		 *
		 * All these gradients must be computed for all examples in the mini-batch, and the resulting 
		 * values should be averaged and then used to update speeds,
		 */
		 
		int iExampleBeginningInInput = 0;
		int iBeginningDeltaOutputRow = 0;
		
		int iOutputDeltaElement = 0;
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
				iOutputDeltaElement = iBeginningDeltaOutputRow + iReceptiveField;
				
				if (dropoutMask[iOutputDeltaElement]) // if output unit is active
				{
					deltaElement = deltaOutputBatch[iOutputDeltaElement];
					
					// Get input value needed, reading it from transpose(input): first move to the correct example, 
					// then pick the right "row and column" using the pre-constructed receptive field lookup table
					iInput = iExampleBeginningInInput + recFieldsLookupTable[iInputTransposeColumn + iReceptiveField];
					
					// Multiply & cumulate in gradientWeight, and also add gradient of L2 penalizer term
					gradientWeight = fma(deltaElement, inputBatch[iInput], gradientWeight) + thisWeightDecay;
					
					// Once per filter, also cumulate error signals in gradientBias
					if (iElement == 0)
					{
						gradientBias += deltaElement;
					}
				}
			}
		}
		
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


/* 
 * ConvUpdateParameters()
 * Updates weights and biases using pre-calculated update speeds (momentum update rule).
 */

__kernel void 
ConvUpdateParameters(	__global float * w,				// arg 0
						__global float * b, 			// arg 1
						__global float * wSpeed, 		// arg 2
						__global float * bSpeed, 		// arg 3
						const int nFilters,				// arg 4
						const int receptiveFieldSize
					)
					
{
	const int iFilter = get_global_id(0); // index of output row (corresponds to one filter)
	const int iElement = get_global_id(1); // index of output col (corresponds to an element of receptive field iFilter)
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix, The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD),
	
	if(iFilter < nFilters && iElement < receptiveFieldSize)
	{
		const int iWeight = iFilter * receptiveFieldSize + iElement;
		const int iBias = iFilter;
		
		w[iWeight] += wSpeed[iWeight];
		
		if (iElement == 0) // this must be done only once per filter
		{
			b[iBias] += bSpeed[iBias];
		}
	}
}

__kernel void 
ConvConstrainWeightNorm(__global float * w,				// arg 0
						const int nFilters,				// arg 4
						const int receptiveFieldSize,	// arg 5
						const float weightMaxNorm
					)
					
{
	const int iFilter = get_global_id(0);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix, The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD),
	
	if(iFilter < nFilters)
	{
		float weightVectorNorm = 0.0F;
		
		for (int iWeightElement = 0; iWeightElement < receptiveFieldSize; iWeightElement++)
		{
			weightVectorNorm += pow(w[iFilter * receptiveFieldSize + iWeightElement], 2);
		}
		weightVectorNorm = sqrt(weightVectorNorm);
		
		if (weightVectorNorm > weightMaxNorm)
		{
			float rescalingFactor = weightMaxNorm / receptiveFieldSize;
			for (int iWeightElement = 0; iWeightElement < receptiveFieldSize; iWeightElement++)
			{
				w[iFilter * receptiveFieldSize + iWeightElement] *= rescalingFactor;
			}
		}
	}
}

