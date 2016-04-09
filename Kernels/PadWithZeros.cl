/* 
 * OpenCL kernel for zero-padding an input vector.
 * Entries of the input arrays are rewritten to a new position, accounting for padding.
 *
 * For global/local work size (both 2 dimensional), use the following settings:
 * 		global_work_size[0] = miniBatchSize;
 *		global_work_size[1] = minimum multiple of WAVEFRONT larger that inputVolume = inputDepth*inputHeight*inputWidth
 * 		local_work_size[0] = miniBatchSize;
 *		local_work_size[1] = WAVEFRONT;
 *
 * WAVEFRONT is a constant value, multiple of 2. Suggested: 32 or 64.
 */
 
 
 __kernel void PadWithZeros (__read_only __global float* input,
							__write_only __global float* paddedInput,
							const int inputWidth, // only Height equal to Width is supported
							const int inputArea,
							const int inputVolume,
							const int outputVolume,
							const int padding,
							const int topRowsOfZeros,
							const int zerosPerSlice
                           )
{	
	const uint iExample = get_global_id(0); // index of current example of mini-batch
	const uint iInInputExample = get_global_id(1); // index of input element within current example
	
	if (iInInputExample < inputVolume) // this is important because of how local_work_size is set (more efficient)
	{
		const uint iSlice = (iInInputExample % inputVolume) / inputArea; // find index of channel within an input volume
		const uint iRow = (iInInputExample % inputArea) / inputWidth; // find index of row within an input channel
		const uint iOutput =  outputVolume*iExample + zerosPerSlice*iSlice + topRowsOfZeros + padding * (2*iRow + 1) + iInInputExample;

		paddedInput[iOutput] = input[iInInputExample + inputVolume*iExample];
	}

    
}
