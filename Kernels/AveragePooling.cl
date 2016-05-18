__kernel void 
AveragePoolingForward(	__global float * output,
						__global float * input,
						const int inputVolume,
						const int inputArea,
						const int nFeatureMaps,
						const int miniBatchSize
			)
{
	const int iExample = get_global_id(0);
	const int iFeatureMap = get_global_id(1);
	
	// Because of how the work sizes are set, the global work size can be larger than the output array, 
	// therefore it is important to check that global indexes are within the array. The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iExample < miniBatchSize && iFeatureMap < nFeatureMaps)
	{
		const int iExampleBeginning = iExample * inputVolume;
		const int iMapBeginning = iFeatureMap * inputArea;
		
		const int iBeginning = iExampleBeginning + iMapBeginning;
		
		float average = 0.0F;
		
		for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
		{
			average += input[iBeginning + iWithinMap];
		}			
		average /= inputArea;
		
		int iOutputActivation = iExample * nFeatureMaps + iFeatureMap;
		output[iOutputActivation] = average;
	}
}


__kernel void 
AveragePoolingBackward(	__global float * deltaInput,
						__global float * deltaOutput,
						const int inputVolume,
						const int inputArea,
						const int nFeatureMaps,
						const int miniBatchSize
					)
{
	const int iExample = get_global_id(0);
	const int iUnit = get_global_id(1);
	
	// Because of how the work sizes are set, the global work size can be larger than the output array, 
	// therefore it is important to check that global indexes are within the array. The computational cost 
	// of this comparison is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iExample < miniBatchSize && iUnit < inputVolume)
	{
		int iFeatureMap = iUnit / inputArea;
		
		float outputGradient = deltaOutput[iExample * nFeatureMaps + iFeatureMap];
		float inputGradient = outputGradient / inputArea;
		
		int iInputActivation = iExample * inputVolume + iUnit;
		
		deltaInput[iInputActivation] = inputGradient;
	}
}