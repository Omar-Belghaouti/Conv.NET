__kernel void 
RandomKernel(	__global float * array,
				const int arrayLength,
				const ulong randomSeed
				)
{

	const int i = get_global_id(0);
	
	
	if(i < arrayLength)
	{
		// randomSeed is a ulong "global" passed to kernel
		ulong thisSeed = randomSeed + i;
		thisSeed = (thisSeed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
		uint pseudoRandomInt = thisSeed >> 16;
		for (int j = 0; j < 5; ++j)
		{
			thisSeed = pseudoRandomInt;
			thisSeed = (thisSeed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
			pseudoRandomInt = thisSeed >> 16;
		}
		array[i] = (float)pseudoRandomInt/(float)4294967295;
	}

}