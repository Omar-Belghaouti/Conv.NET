__kernel void 
CreatePaddingLookupTable (	__global int* paddingLookupTable,
							const int unpaddedWidth, // only Height equal to Width is supported
							const int unpaddedDepth,
							const int padding
						)
{
	const int iUnpadded = get_global_id(0); // index of input element within current example
	
	// 0. Compute some useful quantities
		const int unpaddedArea = unpaddedWidth * unpaddedWidth;
		const int unpaddedVolume = unpaddedDepth * unpaddedArea;
		const int nZerosTopRows = padding * (2 * padding + unpaddedWidth);
		const int nZerosPerChannel = 4 * padding * (unpaddedWidth + padding);
		
	if (iUnpadded < unpaddedVolume) // this is important because of how local/global work sizes are set (more efficient)
	{	
		// 1. Initialize iPadded equal to iUnpadded
		int iPadded = iUnpadded;
		
		// 2. Find index of unpadded slice/channel that we are working on...
		int iChannel = (iUnpadded % unpaddedVolume) / unpaddedArea;
		/// ...and add the number of zeros padding all channels before this one
		iPadded += nZerosPerChannel * iChannel;
		
		// 3. Find index of row that we are working on...
		int iRow = (iUnpadded % unpaddedArea) / unpaddedWidth;
		// ...and add the number of zeros padding all rows before this one
		iPadded += nZerosTopRows + padding * (2*iRow + 1);
		
		// 4. Finally, write the resulting index in the lookup table
		paddingLookupTable[iUnpadded] = iPadded;
	}

}
