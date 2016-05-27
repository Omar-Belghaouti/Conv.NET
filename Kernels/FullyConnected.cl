
__kernel void 
FCForward(	__global float * outputBatch,
			__global float * inputBatch, 
			__global float * weights, 	
			__global float * biases, 	
			const int nInputUnits, 		
			const int nOutputUnits, 
			const int miniBatchSize,
			const float dropoutParameter,
			const ulong randomSeed,
			__global bool * dropoutMask
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
		if (dropoutParameter < 1.0F)
		{
			// generate a pseudo-random number here, mimicking Java RNG
			ulong thisSeed = randomSeed + iOutputActivation;
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
			isUnitOn = pseudoRandFloat < dropoutParameter;
		}
		else
		{
			isUnitOn = true;
		}
		// save unit state
		dropoutMask[iOutputActivation] = isUnitOn;
		
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




__kernel void 
FCBackward(	__global float * deltaXbatch,
			__global float * deltaYbatch,
			__global float * weights,
			__global bool * dropoutMask,
			const int nInputUnits, 		
			const int nOutputUnits, 
			const int miniBatchSize	
				)
{

	const int iInputUnit = get_global_id(0);
	const int iMiniBatchItem = get_global_id(1);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iInputUnit < nInputUnits && iMiniBatchItem < miniBatchSize)
	{
		int iMiniBatchStart = iMiniBatchItem*nOutputUnits;
		int iOutputActivation = 0;
		float sum = 0.0F;
		
		for(int iOutputUnit = 0; iOutputUnit < nOutputUnits; iOutputUnit++)
		{
			iOutputActivation = iMiniBatchStart + iOutputUnit;
			
			if (dropoutMask[iOutputActivation] == true) // if unit is active
			{
				// Multiply & cumulate in sum
				sum += weights[iOutputUnit * nInputUnits + iInputUnit] * deltaYbatch[iOutputActivation];
			}
		}
		
		deltaXbatch[iMiniBatchItem*nInputUnits + iInputUnit] = sum;
	}

}





/* 
 * OpenCL kernel for updating weights/biases speed in FC layers
 * using the gradient computed with backpropagation, for a mini-batch
 * of inputs / delta signals.
 */

__kernel void 
FCUpdateSpeeds(	__global float * wSpeed,	
				__global float * bSpeed,
				__global float * wGradients,	
				__global float * bGradients,
				__global float * inputBatch,	
				__global float * deltaYbatch,
				__global bool * dropoutMask,
				const int nInputUnits,		
				const int nOutputUnits,		
				const float momCoeff,		
				const float learningRate,		
				const int miniBatchSize,
				__global float * weights,
				const float weightDecayCoeff
				)
					
{
	const int iOutput = get_global_id(0);
	const int iInput = get_global_id(1);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(iOutput < nOutputUnits && iInput < nInputUnits) 
	{
		const int iWeight = iOutput * nInputUnits + iInput;
		const float thisWeightDecay = weightDecayCoeff*weights[iWeight];
		const bool isFirstInputUnit = iInput == 0;
		
		float gradientWeight = 0.0;
		float gradientBias = 0.0;
		
		int iOutputElement = 0;
		
		float deltaYElement = 0.0F;
		
		for (int iMiniBatchItem = 0; iMiniBatchItem < miniBatchSize; iMiniBatchItem++)
		{
			iOutputElement = iMiniBatchItem * nOutputUnits + iOutput;
			
			if (dropoutMask[iOutputElement]) // if unit is active
			{
				// Get deltaY element
				deltaYElement = deltaYbatch[iOutputElement];
				
				// Multiply and cumulate in weights gradient, and also add gradient of L2 regularizer term
				gradientWeight += inputBatch[iMiniBatchItem * nInputUnits + iInput] * deltaYElement + thisWeightDecay; 
				
				
				// Just cumulate in biases gradient (but only do this once per output unit!)
				if (isFirstInputUnit)
				{
					gradientBias += deltaYElement;
				}
			}
		}
		
		// Save weight gradient
		wGradients[iWeight] = gradientWeight;
		
		// Update weight speed
		wSpeed[iWeight] = (momCoeff * wSpeed[iWeight]) - learningRate * gradientWeight;
		
		// Update biases speed (once per output unit)
		if (isFirstInputUnit)
		{
			bGradients[iOutput] = gradientBias;
			bSpeed[iOutput] = (momCoeff * bSpeed[iOutput]) - learningRate * gradientBias;
		}
	}
	
}



__kernel void 
FCUpdateParameters(	__global float * w,			
					__global float * b, 		
					__global float * wSpeed, 	
					__global float * bSpeed,
					const int nInput,	
					const int nOutput
					)
					
{
	const int i = get_global_id(0);
	const int j = get_global_id(1);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix. The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD).
	
	if(i < nOutput && j < nInput)
	{
		int iWeight = i*nInput + j;
		w[iWeight] += wSpeed[iWeight];
		
		if (j == 0) // this should be done once per output unit, NOT nInput times!
		{
			b[i] += bSpeed[i];
		}
	}
}

__kernel void 
FCConstrainWeightNorm(__global float * w,				// arg 0
						const int nOutputUnits,			// arg 4
						const int nInputUnits,	// arg 5
						const float weightMaxNorm
					)
					
{
	const int iOutputUnit = get_global_id(0);
	
	// Because of how the local work sizes is set, the global work size can be larger than the output matrix, 
	// therefore it is important to check that global indexes are within the matrix, The computational cost 
	// of these comparisons is greatly compensated by the increased efficiency of using a local work size
	// that is a multiple of WARP (Nvidia) / WAVEFRONT (AMD),
	
	if(iOutputUnit < nOutputUnits)
	{
		float weightVectorNorm = 0.0F;
		
		for (int iWeightElement = 0; iWeightElement < nInputUnits; iWeightElement++)
		{
			weightVectorNorm += pow(w[iOutputUnit * nInputUnits + iWeightElement], 2);
		}
		weightVectorNorm = sqrt(weightVectorNorm);
		
		if (weightVectorNorm > weightMaxNorm)
		{
			float rescalingFactor = weightMaxNorm / nInputUnits;
			for (int iWeightElement = 0; iWeightElement < nInputUnits; iWeightElement++)
			{
				w[iOutputUnit * nInputUnits + iWeightElement] *= rescalingFactor;
			}
		}
	}
}