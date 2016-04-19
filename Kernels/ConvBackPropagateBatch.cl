/* 
 * OpenCL kernel for gradient backpropagation in convolutional layers (deltaY to deltaX)
 * implemented as a matrix multiplication between transpose(weights) and deltaY. Results 
 * are written directly into deltaX (and not into a deltaReceptiveFields matrix), using
 * the pre-computed lookup table. For this reason, it is IMPORTANT to remember to wipe off
 * deltaX (write zeros) before calling this kernel. 
 * All of this is done in parallel across a mini-batch of output gradients.
 */

__kernel void 
ConvBackPropagateBatch(	__global float * deltaInputBatch,
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
		const int iElement = iRow % receptiveFieldSize;
		
		const int iExampleBeginningInDeltaOutput = iExample * (nFilters * nReceptiveFields);
		
		float tmpDeltaInput = 0.0;
		
		for(int iFilter = 0; iFilter < nFilters; iFilter++)
		{
			// Get filter element
			float filterElement = weights[iFilter * receptiveFieldSize + iElement];
			
			// Get error signal corresponding to this mini-batch item, this filter, and this receptiveField:
			// first move to the beginning of this example, then pick the right "row and column"
			int iDeltaOutput = iExampleBeginningInDeltaOutput + iFilter * nReceptiveFields + iReceptiveField;
			float deltaElement = deltaOutputBatch[iDeltaOutput];
			
			// Multiply & cumulate in tmpDeltaInput
			tmpDeltaInput += filterElement * deltaElement;
		}
		
		// Now cumulate this piece of gradient into the correct position of paddedDeltaX (using lookup table)
		// This way, error signals coming from different receptive field positions (but corresponding to the
		// same input position) will be summed, as it should be. (Can be proven on paper.)
		const int iExampleBeginningInDeltaInput = iExample * inputVolume;
		const int iFromLookupTable = recFieldslookupTable[iElement * nReceptiveFields + iReceptiveField];
		const int iDeltaInput = iExampleBeginningInDeltaInput + iFromLookupTable;
		deltaInputBatch[iDeltaInput] += tmpDeltaInput;
		
	}
	
}

