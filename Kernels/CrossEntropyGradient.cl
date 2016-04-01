__kernel void 
CrossEntropyGradient(	__global float* gradient,
						__global float* networkOutput,
						__global float* trueLabelArrays
                )
{
	uint i = get_global_id(0);	
	
	gradient[i] = networkOutput[i] - trueLabelArrays[i];
}