__kernel void 
FCForwardParallel(	__global float * outputBatch,	// arg 0
					__global float * inputBatch, 	// arg 1
					__global float * weights, 		// arg 2
					__global float * biases, 		// arg 3
					const int nInputUnits, 			// arg 4
					const int nOutputUnits, 		// arg 5
					const int miniBatchSize,		// arg 6
					const float dropoutParameter,	// arg 7
					const ulong randomSeed			// arg 8
				)
{

	const int iOutputUnit = get_global_id(0);
	const int iMiniBatchItem = get_global_id(1);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iOutputUnit < nOutputUnits && iMiniBatchItem < miniBatchSize)
	{
		int iOutputActivation = iMiniBatchItem*nOutputUnits + iOutputUnit;
		
		// Dropout here (more efficient: no matrix multiplication if unit is deactivated)
		bool isUnitOn;
		if (dropoutParameter < 1)
		{
			// generate a pseudo-random number here
			ulong thisSeed = randomSeed + iOutputActivation;
			thisSeed = (thisSeed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
			uint pseudoRandomInt = thisSeed >> 16;
			for (int j = 0; j < 5; ++j)
			{
				thisSeed = pseudoRandomInt;
				thisSeed = (thisSeed * 0x5DEECE66DL + 0xBL) & ((1L << 48) - 1);
				pseudoRandomInt = thisSeed >> 16;
			}
			float pseudoRandFloat = (float)pseudoRandomInt/(float)4294967295;
			// this is not a very good pseudo random number, but hopefully it's good enough
			isUnitOn = pseudoRandFloat < dropoutParameter;
		}
		else
		{
			isUnitOn = true;
		}
		
		if (isUnitOn)
		{
			int iMiniBatchStart = iMiniBatchItem * nInputUnits;
			
			float sum = 0.0;
			
			for(int iInputUnit = 0; iInputUnit < nInputUnits; iInputUnit++)
			{
				// Get weight element
				float weightElement = weights[iOutputUnit * nInputUnits + iInputUnit];
				
				// Get input element
				float inputElement = inputBatch[iMiniBatchStart + iInputUnit];

				// Multiply & cumulate in sum
				sum += weightElement * inputElement;
			}
			
			// Add bias
			sum += biases[iOutputUnit];
			
			outputBatch[iOutputActivation] = sum/dropoutParameter;
		}
		else // unit is dropped out
		{
			outputBatch[iOutputActivation] = 0.0f;
		}
	}

}