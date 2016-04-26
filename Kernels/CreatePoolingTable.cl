__kernel void 
CreatePoolingTable(	__global int * poolingTable,		
					const int stride,
					const int inputWidth,
					const int outputWidth
			)
{

	const int i = get_global_id(0); // index of output activation
	
	// Because of how the work sizes are set, the global work size can be larger than the output array, 
	// therefore it is important to check that global indexes are within the array. The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	const int outputArea = outputWidth * outputWidth;
	
	if(i < outputArea)
	{
		int iOutputRow = i / outputWidth;
		int iOutputCol = i % outputWidth;
		
		int iInputTopLeft = iOutputRow * stride * inputWidth + iOutputCol * stride;
		int iInputTopRight = iInputTopLeft + 1;
		int iInputBottomLeft = iInputTopLeft + inputWidth;
		int iInputBottomRight = iInputBottomLeft + 1;
		
		poolingTable[4*i+0] = iInputTopLeft;
		poolingTable[4*i+1] = iInputTopRight;
		poolingTable[4*i+2] = iInputBottomLeft;
		poolingTable[4*i+3] = iInputBottomRight;
		
	}
}