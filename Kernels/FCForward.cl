__kernel void 
FCForward(	__global float * y, // arg 0
			__global float * x, // arg 1
			__global float * W, // arg 2
			__global float * b, // arg 3
			int nInput, 		// arg 4
			int nOutput) 		// arg 5
{

	int i=get_global_id(0);
	
	if(i < nOutput) // not necessary if global work size is set correctly (negligible, however) 
	{
		float value = 0.0;
		
		for(int k = 0; k < nInput; k++)
		{
				value += W[i*nInput + k] * x[k];
		}
		value += b[i];
		
		y[i] = value;
	}

}
