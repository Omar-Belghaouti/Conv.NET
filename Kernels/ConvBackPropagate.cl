/* OpenCL kernel for gradient backpropagation in convolutional layers (deltaY to deltaX)
 * implemented as a matrix multiplication between transpose(weights) and deltaY. Results 
 * are written directly into deltaX (and not into a deltaReceptiveFields matrix), using
 * the pre-computed lookup table. For this reason, it is important to remember to wipe off
 * deltaX (write zeros) before calling this kernel.
 */

__kernel void 
ConvBackPropagate(	__global float * paddedDeltaX,
					__global float * deltaY,
					__global float * w,
					__global int * lookupTable,
					const int nFilters, 		
					const int receptiveFieldSize,
					const int nReceptiveFields
			)
{

	const int iElement = get_global_id(0); // index of output row (corresponds to a receptive field element)
	const int iReceptiveField = get_global_id(1); // index of output col (corresponds to a receptive field)
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iElement < receptiveFieldSize && iReceptiveField < nReceptiveFields)
	{
		float tmpDeltaX = 0.0;
		
		for(int iFilter = 0; iFilter < nFilters; iFilter++)
		{
			// Get filter element from transpose of w
			float filterElement = w[iFilter * receptiveFieldSize + iElement];
			
			// Get error signal corresponding to this filter and this receptiveField
			float deltaElement = deltaY[iFilter * nReceptiveFields + iReceptiveField];
			
			// Multiply & cumulate in gradW
			tmpDeltaX += filterElement * deltaElement;
		}
		
		// Now cumulate this in correct place of paddedDeltaX (using lookup table)
		int locationInInput = lookupTable[iElement * nReceptiveFields + iReceptiveField];
		paddedDeltaX[locationInInput] += tmpDeltaX;
	}
}

