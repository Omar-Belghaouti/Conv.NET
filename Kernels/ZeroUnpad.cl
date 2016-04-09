/* 
 * OpenCL kernel for UNpadding vector "paddedVector".
 *
 * For global/local work size (1 dimensional), use the following settings:
 * 		global_work_size[0] = minimum multiple of WAVEFRONT larger that unpaddedVectorVolume = inputDepth*inputHeight*inputWidth
 * 		local_work_size[0] = WAVEFRONT;
 *
 * WAVEFRONT is a constant value, multiple of 2. Suggested: 32 or 64.
 */
 
 
 __kernel void ZeroUnpad(	__read_only __global float* paddedVector,
							__write_only __global float* unpaddedVector,
							const int inputWidth, // refers to the UNPADDED vector
							const int inputArea, // same
							const int inputVolume, // same
							const int padding,
							const int topRowsOfZeros,
							const int zerosPerSlice
                           )
{	
	const uint iUnpadded = get_global_id(0); // index of input element within current example
	
	if (iUnpadded < inputVolume) // this is important because of how local_work_size is set (more efficient)
	{
		const uint iSlice = (iUnpadded % inputVolume) / inputArea; // find index of channel within an input volume
		const uint iRow = (iUnpadded % inputArea) / inputWidth; // find index of row within an input channel
		const uint iPadded =  zerosPerSlice*iSlice + topRowsOfZeros + padding * (2*iRow + 1) + iUnpadded;

		unpaddedVector[iUnpadded] = paddedVector[iPadded];
	}
}
