/* 
 * OpenCL kernel for UNpadding vector "paddedArrayBatch".
 */
 
 __kernel void 
 ZeroUnpadBatch(__read_only __global float* paddedArrayBatch,
				__write_only __global float* unpaddedArrayBatch,
				const int inputWidth, // refers to the UNPADDED vector
				const int inputArea, // same
				const int inputVolume, // same
				const int padding,
				const int topRowsOfZeros,
				const int zerosPerSlice,
				const int zerosPerVolume
                           )
{	
	const uint iUnpadded = get_global_id(0); // index of input element within current example
		
	if (iUnpadded < inputVolume) // this is important because of how local/global work sizes are set (more efficient)
	{
		// 0. Initialize iPadded equal to iUnpadded
		int iPadded = iUnpadded; 
		
		// 1. Find index of mini-batch item that we are working on...
		const uint iMiniBatchItem = iUnpadded / inputVolume; 
		// ...and add the number of zeros padding all input volumes before this one
		iPadded += zerosPerVolume * iMiniBatchItem;
		
		// 2. Find index of input slice/channel that we are working on...
		const uint iSlice = (iUnpadded % inputVolume) / inputArea;
		/// ...and add the number of zeros padding all channels before this one
		iPadded += zerosPerSlice * iSlice;
		
		// 3. Find index of row that we are working on...
		const uint iRow = (iUnpadded % inputArea) / inputWidth;
		// ...and add the number of zeros padding all rows before this one
		iPadded += topRowsOfZeros + padding * (2*iRow + 1);
		
		// 4. Finally, write this paddedArrayBatch element to correct position in unpaddedArrayBatch
		unpaddedArrayBatch[iUnpadded] = paddedArrayBatch[iPadded];
	}
}
