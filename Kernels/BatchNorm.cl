/* 
 * OpenCL kernel for computing means and averages over spatial dimension AND mini-batch.
 * These values will then be used to normalize input in kernel BatchNormConvForward().
 * New means and average of the mini-batch are also used to update cumulative averages
 * needed for inference. Note that cumulative averages are automatically reset to zero 
 * at the beginning of a new epoch.
 */

 
 // This is going to be VERY slow
__kernel void 
BNConvComputeMeansVariances(__global float * means,
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
	// Global work size = number of feature maps of previous layer = nFilters
	const int iFeatureMap = get_global_id(0);
	
	if(iFeatureMap < inputDepth)
	{
		int iInput = 0;
		
		float mean = 0.0f;
		float variance = 0.0f;
		
		// First mean
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			iInput = iExample * inputVolume + iFeatureMap * inputArea;
			
			for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
			{
				iInput += iWithinMap;
				mean += input[iInput];
			}
		}
		mean /= (miniBatchSize * inputArea);
		
		// save mean and also update cumulative average
		means[iFeatureMap] = mean;
		cumulativeMeans[iFeatureMap] = (iCumulativeAverage * cumulativeMeans[iFeatureMap] + mean) / (iCumulativeAverage + 1);
		
		
		// Now variance
		
		float difference = 0.0f;
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			iInput = iExample * inputVolume + iFeatureMap * inputArea;
			
			for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
			{
				iInput += iWithinMap;
				difference = input[iInput] - mean;
				variance += difference * difference;
			}
		}
		variance /= (miniBatchSize * inputArea);
		
		// Save variance
		variances[iFeatureMap] = variance;
		// and also update cumulative average. Here a corrective factor M/(M+1) is applied to variance
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
BNConvForward(	__global float * output,
				__global float * normalizedInput,
				__global float * input,
				__constant float * means,		// will be over mini-batch if training, running if inference
				__constant float * variances, // same
				__constant float * gammas, 
				__constant float * betas,	
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
		float privateNormalizedInput = (input[iInput] - means[iFeatureMap]) / sqrt(variances[iFeatureMap] + EPSILON);
		normalizedInput[iInput] = privateNormalizedInput;
		
		// Scale and shift
		output[iInput] = gammas[iFeatureMap] * privateNormalizedInput + betas[iFeatureMap];
	}
	
}


__kernel void 
BNConvUpdateSpeeds(	__global float * gammaSpeed,
					__global float * betaSpeed,
					__global float * deltaOutput,
					__global float * normalizedInput,
					const int nParameters,
					const int inputArea,
					const int inputVolume,
					const int miniBatchSize,
					const float momCoeff,
					const float learningRate
				)
{
	// Global work size = number of parameters = inputDepth
	int iParameter = get_global_id(0);
	
	if(iParameter < nParameters)
	{
		// Compute gradients
		
		int iUnit = 0;
		float gammaGrad = 0.0F;
		float betaGrad = 0.0F;
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			iUnit = iExample * inputVolume + iParameter * inputArea; // map beginning
			
			for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
			{
				iUnit += iWithinMap;
				gammaGrad += deltaOutput[iUnit] * normalizedInput[iUnit];
				betaGrad += deltaOutput[iUnit];
			}
		}
		
		// And then update parameter update speed
		gammaSpeed[iParameter] = (momCoeff * gammaSpeed[iParameter]) - learningRate * gammaGrad;
		betaSpeed[iParameter] = (momCoeff * betaSpeed[iParameter]) - learningRate * betaGrad;
	}
	
}


// Good for both convolutional and fully connected layers
__kernel void 
BNUpdateParameters(	__global float * gamma,
					__global float * beta,
					__constant float * gammaSpeed,
					__constant float * betaSpeed,
					const int nParameters
						)
{
	// Global work size = number of parameters = inputDepth
	int iParameter = get_global_id(0);
	
	if(iParameter < nParameters)
	{
		gamma[iParameter] += gammaSpeed[iParameter];
		beta[iParameter] += betaSpeed[iParameter];
	}
	
}


// Compute gradients of loss with respect to mu and sigma^2, needed for backpropagation
// This is probably going to be very slow :(
__kernel void
BNConvGradientMeanVariance( __global float * meanGradient,
							__global float * varianceGradient,
							__global float * deltaOutput,
							__global float * normalizedInput,
							__constant float * gamma,
							__constant float * variance,
							const int inputDepth,
							const int inputArea,
							const int miniBatchSize,
							
							)
{
	// Global work size = 2 * number of parameters = 2 * inputDepth (should be a bit more efficient)
	
	int iFeatureMap = get_global_id(0);							
								
	if (iFeatureMap < 2 * inputDepth)
	{
		const int inputVolume = inputDepth * inputArea;
		
		if (iFeatureMap % 2 == 0) // even indices => compute derivative wrt variance
		{
			iFeatureMap /= 2; // retrieve correct index of feature map
			int iUnit = 0;
			float tmpSumXY = 0.0F;
			
			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				iUnit = iExample * inputVolume + iFeatureMap * inputArea;
			
				for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
				{
					iUnit += iWithinMap;
					tmpSumXY += deltaOutput[iUnit] * normalizedInput[iUnit];
				}
			}
			
			varianceGradient[iFeatureMap] = (-gamma[iFeatureMap] * tmpSumXY) / ( 2*(variance[iFeatureMap] + EPSILON) );
		}
		else // odd indices => compute derivative wrt mean
		{
			iFeatureMap /= 2; // retrieve correct index of feature map
			int iUnit = 0;
			float tmpSumX = 0.0F;
			float tmpSumY = 0.0F;
			float tmpSumXY = 0.0F;
			
			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				iUnit = iExample * inputVolume + iFeatureMap * inputArea;
			
				for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
				{
					iUnit += iWithinMap;
					
					tmpSumX += normalizedInput[iUnit];
					tmpSumY += deltaOutput[iUnit];
					tmpSumXY += deltaOutput[iUnit] * normalizedInput[iUnit];
				}
			}
			
			meanGradient[iFeatureMap] =  (tmpSumX * tmpSumXY / (miniBatchSize * inputArea) - tmpSumY);
			meanGradient[iFeatureMap] *= gamma[iFeatureMap]  / sqrt(variance[iFeatureMap] + EPSILON);
		}
	}		
}
							
							
							

__kernel void
BNConvBackPropagate(__global float * deltaInput,
					__global float * deltaOutput,
					__global float * input,
					__constant float * gamma,
					__constant float * mean,
					__constant float * variance,
					__constant float * meanGradient,
					__constant float * varianceGradient,
					const int inputArea,
					const int inputVolume,
					const int miniBatchSize,
					)
{
	// Global work size = number of activations = tensor volume * mini-batch size
	const int iUnit = get_global_id(0);
	
	if(iUnit < inputVolume * miniBatchSize)
	{
		// Retrieve index of feature map that this input values belongs to
		int iFeatureMap = (iUnit % inputVolume) / inputArea;
		
		float tmp = 0.0F;
		
		// See backprop expression...
		tmp += 2 * varianceGradient[iFeatureMap]* (input[iUnit] - mean[iFeatureMap]) + meanGradient[iFeatureMap];
		tmp /= (miniBatchSize * inputArea); 
		tmp += ( (gamma[iFeatureMap] * deltaOutput[iUnit]) / sqrt(variance[iFeatureMap] + EPSILON) );
		
		// Write gradient
		deltaInput[iUnit] = tmp;
	}
}

