__kernel void 
CreateMaxPoolingTable(	__global int * poolingTable,		
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


__kernel void 
MaxPoolingForward(	__global float * output,
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


__kernel void 
MaxPoolingBackward(__global float * deltaInput,
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