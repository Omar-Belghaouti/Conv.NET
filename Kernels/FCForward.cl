#define TOLERANCE 0.00001

__kernel void 
fcForward(	__global float * W, 
			__global float * x, 
			__global float * b,
			__global float * y,
			int nInputs, 
			int nOutputs) 
{
	int i=get_global_id(0);

	if(i < nOutputs)
	{
		for(int k = 0; k < nInputs; k++)
		{
			if (x[k] > TOLERANCE) // naive sparsity exploit
				C[i]+=A[i*nInputs + k] * x[k];
		}
		C[i] += b[i];
	}
}