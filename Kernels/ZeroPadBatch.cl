/* 
 * OpenCL kernel for zero-padding an input vector containing a mini-batch
 * Entries of the input arrays are rewritten to a new position, accounting for padding.
 */
 
 __kernel void 
 ZeroPadBatch (	__read_only __global float* inputBatch,
				__write_only __global float* paddedInputBatch,
				const int inputWidth, // only Height equal to Width is supported
				const int inputArea,
				const int inputVolume,
				const int padding,
				const int topRowsOfZeros,
				const int zerosPerSlice,
				const int zerosPerVolume
                )
{	
	const uint iInput = get_global_id(0); // index of input element within current example
	
	if (iInput < inputVolume) // this is important because of how local/global work sizes are set (more efficient)
	{
		// 0. Initialize iOutput equal to iInput
		int iOutput = iInput; 
		
		// 1. Find index of mini-batch item that we are working on...
		const uint iMiniBatchItem = iInput / inputVolume; 
		// ...and add the number of zeros padding all input volumes before this one
		iOutput += zerosPerVolume * iMiniBatchItem;
		
		// 2. Find index of input slice/channel that we are working on...
		const uint iSlice = (iInput % inputVolume) / inputArea;
		/// ...and add the number of zeros padding all channels before this one
		iOutput += zerosPerSlice * iSlice;
		
		// 3. Find index of row that we are working on...
		const uint iRow = (iInput % inputArea) / inputWidth;
		// ...and add the number of zeros padding all rows before this one
		iOutput += topRowsOfZeros + padding * (2*iRow + 1);
		
		// 4. Finally, write this inputBatch element to correct position in paddedInputBatch array
		paddedInputBatch[iOutput] = inputBatch[iInput];
	}

}


