__kernel void 
SkipForward(	__global float * output,
				__global float * input, 		
				const int nUnits, 
				const int miniBatchSize
		)
{
	const int iActivation = get_global_id(0);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iActivation < miniBatchSize*nUnits)
	{
		output[iActivation] += input[iActivation];
	}
}

__kernel void 
SkipBackward(	__global float * deltaInput,
				__global float * deltaOutput, 		
				const int nUnits, 
				const int miniBatchSize
		)
{
	const int iActivation = get_global_id(0);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iActivation < miniBatchSize*nUnits)
	{
		deltaInput[iActivation] += deltaOutput[iActivation];
	}
}

// ...as simple as that! :)







/*
__kernel void 
SkipForward(	__global float * output,
				__global float * input, 		
				const int nUnits, 
				const int miniBatchSize
		)
{
	const int iExample = get_global_id(0);
	const int iUnit = get_global_id(1);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iExample < miniBatchSize && iUnit < nUnits)
	{
		int iActivation = iExample * nUnits + iUnit;
		
		output[iActivation] += input[iActivation];
	}
}

__kernel void 
SkipBackward(	__global float * deltaInput,
				__global float * deltaOutput, 		
				const int nUnits, 
				const int miniBatchSize
		)
{
	const int iExample = get_global_id(0);
	const int iUnit = get_global_id(1);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iExample < miniBatchSize && iUnit < nUnits)
	{
		int iActivation = iExample * nUnits + iUnit;
		
		deltaInput[iActivation] += deltaOutput[iActivation];
	}
}
*/
// ...as simple as that! :)