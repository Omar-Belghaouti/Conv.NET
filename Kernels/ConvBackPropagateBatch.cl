/* OpenCL kernel for gradient backpropagation in convolutional layers (deltaY to deltaX)
 * implemented as a matrix multiplication between transpose(weights) and deltaY. Results 
 * are written directly into deltaX (and not into a deltaReceptiveFields matrix), using
 * the pre-computed lookup table. For this reason, it is IMPORTANT to remember to wipe off
 * deltaX (write zeros) before calling this kernel. 
 * All of this is done in parallel for a mini-batch of output gradients.
 */

__kernel void 
ConvBackPropagateBatch(	__write_only __global float * deltaInputBatch,
						__read_only __global float * deltaOutputBatch,
						__read_only __global float * weights,
						__read_only __global int * lookupTable,
						const int nFilters, 		
						const int receptiveFieldSize,
						const int nReceptiveFields,
						const int miniBatchSize
					)
{

	const int iRow = get_global_id(0); // index of output row
	const int iReceptiveField = get_global_id(1); // index of output col (corresponds to a receptive field)
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iRow < receptiveFieldSize * miniBatchSize && iReceptiveField < nReceptiveFields)
	{
		const int iMiniBatchItem = iRow / (receptiveFieldSize * nReceptiveFields);
		const int iElement = iRow % (receptiveFieldSize * nReceptiveFields);
		
		const int iMiniBatchItemBeginningInDeltaOutput = iMiniBatchItem * (nFilters * nReceptiveFields);
		
		float tmpDeltaInput = 0.0;
		
		for(int iFilter = 0; iFilter < nFilters; iFilter++)
		{
			// Get filter element
			float filterElement = weights[iFilter * receptiveFieldSize + iElement];
			
			// Get error signal corresponding to this mini-batch item, this filter, and this receptiveField
			int iDeltaOutput = iMiniBatchItemBeginningInDeltaOutput + iFilter * nReceptiveFields + iReceptiveField;
			float deltaElement = deltaOutputBatch[iDeltaOutput];
			
			// Multiply & cumulate in tmpDeltaX
			tmpDeltaInput += filterElement * deltaElement;
		}
		
		// Now cumulate this in correct position of paddedDeltaX (using lookup table)
		// This way, error signals coming from different receptive field positions (but corresponding to the
		// same input position) will be summed.
		const int iMiniBatchItemBeginningInDeltaInput = iMiniBatchItem * (receptiveFieldSize * nReceptiveFields);
		const int iFromLookupTable = lookupTable[iElement * nReceptiveFields + iReceptiveField];
		const int iDeltaInput = iMiniBatchItemBeginningInDeltaInput + iFromLookupTable;
		deltaInputBatch[iDeltaInput] += tmpDeltaInput;
		
	}
}

