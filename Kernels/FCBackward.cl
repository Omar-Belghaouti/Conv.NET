__kernel void 
FCBackward(	__global float * deltaX,	// arg 0
			__global float * deltaY,	// arg 1
			__global float * W,			// arg 2
			int nInput, 				// arg 3
			int nOutput) 				// arg 4
{
	int i=get_global_id(0);
	
	if(i < nInput) // not necessary if global work size is set correctly (negligible, however)
	{
		float value = 0.0;
		for(int k = 0; k < nOutput; k++)
		{
			value += W[k*nInput + i] * deltaY[k]; 
		}
		
		deltaX[i] = value;
	}
}