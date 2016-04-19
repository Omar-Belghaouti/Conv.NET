/* 
 * OpenCL kernel for updating weights/biases speed in FC layers
 * using the gradient computed with backpropagation, for a mini-batch
 * of inputs / delta signals.
 */

__kernel void 
FCUpdateSpeedsParallel(	__global float * wSpeed,	// arg 0
						__global float * bSpeed,	// arg 1
						__global read_only float * inputBatch,		// arg 2
						__global read_only float * deltaYbatch,	// arg 3
						const int nInputUnits,				// arg 4
						const int nOutputUnits,				// arg 5
						const float momCoeff,				// arg 6
						const float learningRate,			// arg 7
						const int miniBatchSize				// arg 8
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
		const bool isFirstInputUnit = iInput == 0;
		
		float gradientWeight = 0.0;
		float gradientBias = 0.0;
		
		for (int iMiniBatchItem = 0; iMiniBatchItem < miniBatchSize; iMiniBatchItem++)
		{
			// Get input element
			float inputElement = inputBatch[iMiniBatchItem * nInputUnits + iInput];
			
			// Get deltaY element
			float deltaYElement = deltaYbatch[iMiniBatchItem * nOutputUnits + iOutput];
			
			// Multiply and cumulate in weights gradient
			gradientWeight += inputElement * deltaYElement; 
			
			// Just cumulate in biases gradient (but only do this once per output unit!)
			if (isFirstInputUnit)
			{
				gradientBias += deltaYElement;
			}
		}
		
		// Update weight speed
		wSpeed[iWeight] = (momCoeff * wSpeed[iWeight]) - (learningRate/miniBatchSize) * gradientWeight;
		
		// Update biases speed (once per output unit)
		if (isFirstInputUnit)
		{
			bSpeed[iOutput] = (momCoeff * bSpeed[iOutput]) - (learningRate/miniBatchSize) * gradientBias;
		}
	}
	
}