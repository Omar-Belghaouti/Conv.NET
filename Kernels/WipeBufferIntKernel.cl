 __kernel void WipeBufferIntKernel(	__global int* buffer,
									const int nBufferElements
                           )
{	
	const uint i = get_global_id(0);
	
	if (i < nBufferElements) // this is important because of how local_work_size is set (more efficient)
	{
		buffer[i] = 0;
	}
}
