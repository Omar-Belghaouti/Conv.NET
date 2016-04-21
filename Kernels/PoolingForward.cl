__kernel void 
PoolingForward(	__global float * output,
				__global float * input,
				__global bool * switches, // same dim as input
				__global int * poolingTable,
				const int inputVolume,
				const int inputArea,
				const int outputVolume,
				const int outputArea,
				const int miniBatchSize
			)
{
	const int iOutput = get_global_id(0); // index of output activation
	
	// Because of how the work sizes are set, the global work size can be larger than the output array, 
	// therefore it is important to check that global indexes are within the array. The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iOutput < outputVolume * miniBatchSize)
	{
		int iExample = iOutput / outputVolume;
		int iChannel = (iOutput % outputVolume) / outputArea;
		int iOutputWithinChannel = iOutput % outputArea;
		
		int iInputChannelBeginning = iExample * inputVolume + iChannel * inputArea;
		
		// Get input elements to downsample
		int iPoolingField[4];
		float poolingField[4];
		for (int i = 0; i < 4; i++)
		{
			iPoolingField[i] = iInputChannelBeginning + poolingTable[4 * iOutputWithinChannel + i];
			poolingField[i] = input[iPoolingField[i]];
		}			
		
		// Downsample
		float maxInput = -INFINITY;
		for (int i = 0; i < 4; i++)
		{
			if (poolingField[i] > maxInput)
				maxInput = poolingField[i];
		}
		output[iOutput] = maxInput;
		
		// Save indices of switches
		for (int i = 0; i < 4; i++)
		{
			switches[iPoolingField[i]] = (poolingField[i] == maxInput) ? true : false;
		}
		
	}
}