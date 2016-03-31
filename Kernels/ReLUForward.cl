__kernel void 
ReLUForward(__global float * y, // arg 0
			__global float * x, // arg 1
			int nOutput) 		// arg 2
{
	int i=get_global_id(0);
	
	if(i < nOutput) // not necessary if global work size is set correctly (negligible, however) 
	{
		if (x[i] < 0)
			y[i] = 0.0;
		else
			y[i] = x[i];
	}
}