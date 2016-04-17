// can be improved _A LOT_!!

__kernel void 
SoftmaxForward (__global write_only float* outputBatch,
				__global read_only float* inputBatch,
				__global float* auxFloat,
				const int nUnits,
				const int miniBatchSize
                )
{
	
	const int iMiniBatchItem = get_global_id(0);
	const int iUnit = get_global_id(0);	
	
	if (iMiniBatchItem < miniBatchSize && iUnit < nUnits)
	{
		int iMiniBatchItemStart = iMiniBatchItem * nUnits;
		int iOutputArray = iMiniBatchItemStart + iUnit;
		
		// For each mini-batch item, find maximum preactivation
		if (iUnit == 0) {
			float maxInput = inputBatch[iOutputArray];
			for (int j = 1; j < nUnits; j++)
			{
				if (inputBatch[iMiniBatchItemStart + j] > maxInput)
					maxInput = inputBatch[iMiniBatchItemStart + j];
			}
			// write max in auxiliary variable
			auxFloat[iMiniBatchItem] = maxInput;
		}
		
		// Wait for all threads to reach this point
		barrier(CLK_GLOBAL_MEM_FENCE);
		
		// Compute exp(input - maxInput)
		float tmpOutput = exp(inputBatch[iOutputArray] - auxFloat[iMiniBatchItem]);
		
		// Wait for all threads to reach this point
		barrier(CLK_GLOBAL_MEM_FENCE);
		
		// Sum activations
		if (iUnit == 0) {
			float sum = 0.0;
			for (int j = 0; j < nUnits; j++) {
				sum += outputBatch[iMiniBatchItemStart + j];
			}
			// write sum in auxiliary variable
			auxFloat[iMiniBatchItem] = sum;
		}
		
		// Wait for all threads to reach this point
		barrier(CLK_GLOBAL_MEM_FENCE);
		
		// Write output
		outputBatch[iOutputArray] = tmpOutput / auxFloat[iMiniBatchItem];
	}
}