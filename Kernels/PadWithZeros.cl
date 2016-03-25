/* 
 * OpenCL kernel for zero-padding an input vector.
 * Entries of the input arrays are rewritten to a new position, accounting for padding.
 * The global work size should be equal to the length of the INPUT array.
 */
 
__kernel void PadWithZeros (__read_only __global float* input,
							__write_only __global float* paddedInput,
							int padding,
							int inputWidth, // only Height equal to Width is supported
							int inputArea,
							int inputVolume 
                           )
{

    const uint zerosPerSlice = 2 * padding * (inputWidth + inputWidth + 2 * padding);
	const uint shiftForAll = padding + padding * (2 * padding + inputWidth);
	
	const uint k = get_global_id(0);

    const uint iRow = (k % inputArea) / inputWidth;
	const uint iSlice = (k % inputVolume) / inputArea;
	
	const uint iNew = k + shiftForAll + 2*padding*iRow + zerosPerSlice*iSlice;

	paddedInput[iNew] = input[k];
}