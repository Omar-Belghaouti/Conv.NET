/* OpenCL kernel to update weights and biases change speed in FC layers
 * using the gradient computed with backpropagation
 */

__kernel void 
FCUpdateSpeeds(	__global write_only float * wSpeed, 	// arg 0
				__global write_only float * bSpeed, 	// arg 1
				__global read_only float * x,		// arg 2
				__global read_only float * deltaY,	// arg 3
				const int nInput,					// arg 4
				const int nOutput,					// arg 5
				const float momCoeff,				// arg 6
				const float learnRatePerItem,		// arg 7
				const int miniBatchItem				// arg 8
					)
					
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(i < nOutput && j < nInput) 
	{
		const int iWeight = i*nInput + j;
		
		// 1. Update weights speed
		
		float speed = wSpeed[iWeight];
		if (miniBatchItem == 0)
		{
			// speed decay (only applies once when looping through the mini-batch)
			speed *= momCoeff;
		}
		// gradient-based speed update (note that this learning rate is *per mini-batch item*)
		speed -= learnRatePerItem * x[j] * deltaY[i]; 
	
		wSpeed[iWeight] = speed;
		
		// 2. Update biases speed
		if (j == 0) // otherwise it will be updated nInput times!
		{
			speed = bSpeed[i];
			if (miniBatchItem == 0)
			{
				// speed decay (only applies once when looping through the mini-batch)
				speed *= momCoeff;
			}
			// gradient-based speed update (note that this learning rate is *per mini-batch item*)
			speed -= learnRatePerItem * deltaY[i];
		
			bSpeed[i] = speed;
		}
	}
}