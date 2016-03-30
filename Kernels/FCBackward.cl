__kernel void 
FCBackward(	__global float * deltaX,	// arg 0
			__global float * deltaY,	// arg 1
			__global float * W,			// arg 2
			int nInput, 				// arg 3
			int nOutput) 				// arg 4
{
	int i=get_global_id(0);
	deltaX[i] = 0.0;
	
	if(i < nInput) // not necessary if global work size is set correctly (negligible, however)
	{
		for(int k = 0; k < nOutput; k++)
		{
			deltaX[i] += W[k*nInput + i] * deltaY[k]; 
		}
	}
}