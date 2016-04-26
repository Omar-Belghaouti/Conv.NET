__kernel void 
PoolingBackward(__global float * deltaInput,
				__global float * deltaOutput,
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
		int iInputChannelBeginning = iExample * inputVolume + iChannel * inputArea;
		
		int iOutputWithinChannel = iOutput % outputArea;
		
		for (int i = 0; i < 4; i++)
		{
			int iInput = iInputChannelBeginning + poolingTable[4 * iOutputWithinChannel + i];
			if (switches[iInput])
				deltaInput[iInput] = deltaOutput[iOutput];
			else 
				deltaInput[iInput] = 0.0f;
		}		
	}
}