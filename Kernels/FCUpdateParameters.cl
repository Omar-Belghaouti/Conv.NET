__kernel void 
FCUpdateParameters(	__global float * W,			// arg 0
					__global float * b, 		// arg 1
					__global float * W_speed, 	// arg 2
					__global float * b_speed, 	// arg 3
					__global float * x,			// arg 4
					__global float * deltaY,	// arg 5
					const int nInput,			// arg 6
					const int nOutput,			// arg 7
					const float learnRate,		// arg 8
					const float momCoeff		// arg 9
					)
					
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	
	if(i < nOutput && j < nInput) // not necessary if global work size is set correctly (negligible, however) 
	{
		int this_index = i*nInput+j;
		W_speed[this_index] *= momCoeff; // speed decay
		W_speed[this_index] -= learnRate * x[j] * deltaY[i]; // speed gradient-based update
	
		W[this_index] += W_speed[this_index];
		
		if (j == 0) // otherwise it will be updated nInput times
		{
			b_speed[i] *= momCoeff; // speed decay
			b_speed[i] -= learnRate * deltaY[i]; // speed gradient-based update
		
			b[i] += b_speed[i];
		}
	}
}