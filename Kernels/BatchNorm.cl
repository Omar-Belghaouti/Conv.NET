/* 
 * OpenCL kernel for computing means and averages over spatial dimension AND mini-batch.
 * These values will then be used to normalize input in kernel BatchNormConvForward().
 * New means and average of the mini-batch are also used to update cumulative averages
 * needed for inference. Note that cumulative averages are automatically reset to zero 
 * at the beginning of a new epoch.
 */

 
__kernel void 
ComputeMeansVariancesConv(	__global float * means,
							__global float * variances,
							__global float * cumulativeMeans,
							__global float * cumulativeVariances,
							__global float * input, 
							const int inputDepth, // = nFilters 	
							const int inputArea,
							const int inputVolume,
							const int miniBatchSize,
							const int iCumulativeAverage // index of current sample of mu and sigma^2 (should be between 0 and dataSetSize/miniBatchSize)
				)
{
	const int iFeatureMap = get_global_id(0);
	
	if(iFeatureMap < inputDepth)
	{
		float mean = 0.0f;
		float variance = 0.0f;
		
		int iExampleBeginning = 0;
		int iMapBeginning = 0;
		int iInput = 0;
		
		// First mean
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			iExampleBeginning = iExample * inputVolume;
			iMapBeginning = iExampleBeginning + iFeatureMap * inputArea;
			iInput = iMapBeginning;
			
			for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
			{
				iInput += iWithinMap;
				mean += input[iInput];
			}
		}
		means[iFeatureMap] = mean / (miniBatchSize * inputArea);
		
		// and also update cumulative average
		cumulativeMeans[iFeatureMap] = (iCumulativeAverage * cumulativeMeans[iFeatureMap] + mean) / (iCumulativeAverage + 1);
		
		
		// Then variance
		
		float difference = 0.0f;
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			iExampleBeginning = iExample * inputVolume;
			iMapBeginning = iExampleBeginning + iFeatureMap * inputArea;
			iInput = iMapBeginning;
			
			for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
			{
				iInput += iWithinMap;
				difference = input[iInput] - mean;
				variance += difference * difference;
			}
		}
		variances[iFeatureMap] = variance / (miniBatchSize * inputArea);
		
		// and also update cumulative average. Here a corrective factor M/(M+1) is applied to <variance> 
		// before updating cumulative average, in order to produce an unbiased estimate
		variance *= miniBatchSize / (miniBatchSize + 1);		
		cumulativeVariances[iFeatureMap] = (iCumulativeAverage * cumulativeVariances[iFeatureMap] + variance) / (iCumulativeAverage + 1);
		
	}
}


/* 
 * OpenCL kernel for forward pass in BatchNorm layer following a Conv layer.
 * Input activations are first normalized using either mean and variance computed
 * over the current mini-batch (if we are training) or the running statistics computed
 * over the last training epoch (if we are doing inference).
 * Then, these normalized values are scaled and shifted using learnable parameters
 * gamma and beta (one pair of parameters per feature map).
 */
 
 #define EPSILON 1.0E-6
 
__kernel void 
BatchNormConvForward(	__global float * output,
						__global float * input,
						__global float * means,		// will be over mini-batch if training, running if inference
						__global float * variances, // same
						__global float * gammas, 
						__global float * betas,	
						const int inputArea,
						const int inputVolume,
						const int miniBatchSize
				)
{
	// Global work size = number of activations = tensor volume * mini-batch size
	const int iInput = get_global_id(0);
	
	if(iInput < inputVolume * miniBatchSize)
	{
		// Retrieve index of feature map that this input values belongs to
		int iFeatureMap = (iInput % inputVolume) / inputArea;
		
		// Normalize input, using the pre-calculated mean and variance
		float normalizedInput = (input - means[iFeatureMap]) / sqrt(variances[iFeatureMap] + EPSILON);
		
		// Scale and shift
		output[iInput] = gammas[iFeatureMap] * normalizedInput + betas[iFeatureMap];
	}
	
}


