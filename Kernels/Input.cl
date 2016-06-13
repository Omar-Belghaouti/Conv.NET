__kernel void
Dropout( __global float * activations,
	const int nActivations,
	const float dropoutParameter, // probability of keeping the unit ALIVE
	const ulong randomSeed
	)
{
	const int iActivation = get_global_id(0);

	if (iActivation < nActivations)
	{
		// generate a pseudo-random number here, mimicking Java RNG
		ulong thisSeed = randomSeed + iActivation;
		thisSeed = (thisSeed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
		uint pseudoRandomInt = thisSeed >> 16;
		for (int j = 0; j < 6; ++j)
		{
			thisSeed = pseudoRandomInt;
			thisSeed = (thisSeed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
			pseudoRandomInt = thisSeed >> 16;
		}
		float pseudoRandFloat = (float)pseudoRandomInt/(float)4294967295;
		// this is not a very good pseudo random number, but hopefully it's good enough
		
		if (pseudoRandFloat > dropoutParameter) // unit is dropped
			activations[iActivation] = 0.0F;
		else
			activations[iActivation] /= dropoutParameter;
	}
}
