__kernel void 
ReLUBackward(	__global float * deltaX, 	// arg 0
				__global float * deltaY, 	// arg 1
				__global float * x,			// arg 2
				int nInput) 				// arg 3
{
	int i=get_global_id(0);
	
	if(i < nInput) // not necessary if global work size is set correctly (negligible, however) 
	{
		if (x[i] < 0)
			deltaX[i] = 0.0;
		else
			deltaX[i] = deltaY[i];
	}
}