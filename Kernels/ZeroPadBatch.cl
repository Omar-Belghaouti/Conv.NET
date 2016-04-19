/* 
 * OpenCL kernel for zero-padding an input vector containing a mini-batch
 * Entries of the input arrays are rewritten to a new position, accounting for padding.
 */
 
 __kernel void 
 ZeroPadBatch (	__global float* paddedInputBatch,
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
		int iUnpaddedWithinExample = iUnpadded % unpaddedVolume;
		
		// Find beginning of mini-batch item that we are working on in *padded* array
		int iExampleBeginningInPadded = iExample * paddedVolume;
		
		// Find index of destination element (in padded array) using the lookup table
		int iPadded = iExampleBeginningInPadded + paddingLookupTable[iUnpaddedWithinExample];
		
		// Write value
		paddedInputBatch[iPadded] = inputBatch[iUnpadded];
	}
}


