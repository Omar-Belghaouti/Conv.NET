/*
 * OpenCL kernels used by BatchNormConv layer.
 *	- BNConvComputeMeansVariances;
 *	- BNConvForward;
 *	- BNConvBackPropagate;
 *	- BNConvGradientMeanVariance;  (auxiliary for BackPropagate)
 *	- BNConvUpdateSpeeds;
 *	- BNUpdateParameters;
 */
 
#define EPSILON 1.0E-6 // constant small number needed to ensure not to divide by zero when dividing by standard deviation



/**********************************
 * TODO: get rid of extra kernels
 **********************************/


/* ==================================================================================================================================== */

/* BNCONVCOMPUTEMEANSVARIANCES()
 * Computes means and averages over spatial dimension AND mini-batch (Conv case).
 * These values will then be used to normalize input in kernel BatchNormConvForward().
 * New means and average of the mini-batch are also used to update cumulative averages
 * needed for inference. Note that cumulative averages are automatically reset to zero 
 * at the beginning of a new epoch.
 */

 
 // WARNING: This may be VERY slow...
 
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
		int iUnit = 0;
		
		float mean = 0.0f;
		float variance = 0.0f;
		
		// First mean
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			iUnit = iExample * inputVolume + iFeatureMap * inputArea;
			
			for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
			{
				iUnit += iWithinMap;
				mean += input[iUnit];
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
			iUnit = iExample * inputVolume + iFeatureMap * inputArea;
			
			for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
			{
				iUnit += iWithinMap;
				difference = input[iUnit] - mean;
				variance += difference * difference;
			}
		}
		variance /= (miniBatchSize * inputArea - 1);
		
		// Save variance and update cumulativeVariance
		variances[iFeatureMap] = variance;		
		cumulativeVariances[iFeatureMap] = (iCumulativeAverage * cumulativeVariances[iFeatureMap] + variance) / (iCumulativeAverage + 1);
	}
}

/* ==================================================================================================================================== */

/* BNFCCOMPUTEMEANSVARIANCES()
 * Computes means and averages over the mini-batch (FC case).
 * These values will then be used to normalize input in kernel BNFCForward().
 * New means and average of the mini-batch are also used to update cumulative averages
 * needed for inference. Note that cumulative averages are automatically reset to zero 
 * at the beginning of a new epoch.
 */

 
// WARNING: This may be VERY slow...

__kernel void 
BNFCComputeMeansVariances(	__global float * means,
							__global float * variances,
							__global float * cumulativeMeans,
							__global float * cumulativeVariances,
							__global float * input, 
							const int nUnits,
							const int miniBatchSize,
							const int iCumulativeAverage // index of current sample of mu and sigma^2 (should be between 0 and dataSetSize/miniBatchSize)
				)
{
	// Global work size = number of statistics to compute = number of units in layer
	const int iStatistics = get_global_id(0);
	
	if(iStatistics < nUnits)
	{
		// First compute  mean
		
		int iUnit = iStatistics;
		
		float mean = 0.0F;
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			mean += input[iUnit];
			
			iUnit += nUnits;
		}
		mean /= miniBatchSize;
		
		// save mean and also update cumulative average
		means[iStatistics] = mean;
		cumulativeMeans[iStatistics] = (iCumulativeAverage * cumulativeMeans[iStatistics] + mean) / (iCumulativeAverage + 1);
		
		
		// Now compute variance
		
		iUnit = iStatistics; // reset index
		
		float difference = 0.0F;
		float variance = 0.0F;
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			difference = input[iUnit] - mean;
			variance += (difference * difference);
			
			iUnit += nUnits;
		}
		variance /= (miniBatchSize - 1);
		
		// Save variance and update cumulativeVariance
		variances[iStatistics] = variance;		
		cumulativeVariances[iStatistics] = (iCumulativeAverage * cumulativeVariances[iStatistics] + variance) / (iCumulativeAverage + 1);
		
	}
}

/* ==================================================================================================================================== */

/* BNCONVFORWARD()
 * OpenCL kernel for forward pass in BatchNorm layer following a Conv layer.
 * Input activations are first normalized using either mean and variance computed
 * over the current mini-batch (if we are training) or the cumulative statistics computed
 * over the last training epoch (if we are doing inference).
 * Then, these normalized values are scaled and shifted using learnable parameters
 * gamma and beta (one pair of parameters per feature map).
 */
 
__kernel void 
BNConvForward(	__global float * output,
				__global float * normalizedInput,
				__global float * input,
				__constant float * means,		// will be over mini-batch if training, cumulative if inference
				__constant float * variances, 	// same
				__constant float * gammas, 
				__constant float * betas,	
				const int inputArea,
				const int inputVolume,
				const int miniBatchSize
				)
{
	// Global work size = number of activations = tensor volume * mini-batch size
	const int iUnit = get_global_id(0);
	
	if(iUnit < inputVolume * miniBatchSize)
	{
		// Retrieve index of feature map that this input values belongs to
		int iFeatureMap = (iUnit % inputVolume) / inputArea;
		
		// Normalize input, using the pre-calculated mean and variance
		float privateNormalizedInput = (input[iUnit] - means[iFeatureMap]) / sqrt(variances[iFeatureMap] + EPSILON);
		normalizedInput[iUnit] = privateNormalizedInput;
		
		// Scale and shift
		output[iUnit] = gammas[iFeatureMap] * privateNormalizedInput + betas[iFeatureMap];
	}
}

/* ==================================================================================================================================== */

/* BNFCFORWARD()
 * OpenCL kernel for forward pass in BatchNorm layer following a FC layer.
 * Input activations are first normalized using either mean and variance computed
 * over the current mini-batch (if we are training) or the cumulative statistics computed
 * over the last training epoch (if we are doing inference).
 * Then, these normalized values are scaled and shifted using learnable parameters
 * gamma and beta (one pair of parameters per feature map).
 */
 
__kernel void 
BNFCForward(__global float * output,
			__global float * normalizedInput,
			__global float * input,
			__constant float * means,		// will be over mini-batch if training, cumulative if inference
			__constant float * variances, 	// same
			__constant float * gammas, 
			__constant float * betas,	
			const int nUnits,
			const int miniBatchSize
			)
{
	// Global work size = number of activations = nUnits * mini-batch size
	const int iUnit = get_global_id(0);
	
	if(iUnit < nUnits * miniBatchSize)
	{
		// Retrieve index of mean / variance corresponding to this unit
		int iStatistics = iUnit % nUnits;
		
		// Normalize input, using the pre-calculated mean and variance
		float privateNormalizedInput = (input[iUnit] - means[iStatistics]) / sqrt(variances[iStatistics] + EPSILON);
		normalizedInput[iUnit] = privateNormalizedInput;
		
		// Scale and shift
		output[iUnit] = gammas[iStatistics] * privateNormalizedInput + betas[iStatistics];
	}
}

/* ==================================================================================================================================== */

/* BNCONVUPDATESPEEDS()
 * Computes gradients of loss function with respect to learnable parameters gamma and beta
 * in case previous layer is Convolutional. The gradients are then used to update parameter
 * change speed according to momentum rule and learning rate.
 */

__kernel void 
BNConvUpdateSpeeds(	__global float * gammaSpeed,
					__global float * betaSpeed,
					__global float * deltaOutput,
					__global float * normalizedInput,
					const int inputDepth,
					const int inputArea,
					const int inputVolume,
					const int miniBatchSize,
					const float momCoeff,
					const float learningRate
				)
{
	// Global work size = number of parameters = 2 * inputDepth (should be a bit more efficient)
	const int i = get_global_id(0);
	
	if(i < 2 * inputDepth)
	{
		if (i % 2 == 0) // even indices => gradient wrt gamma
		{
			int iParameter = i /2; // retrieve correct index of parameter

			int iUnit = 0;
			float gammaGrad = 0.0F;

			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				iUnit = iExample * inputVolume + iParameter * inputArea; // map beginning
				
				for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
				{
					iUnit += iWithinMap;
					gammaGrad += deltaOutput[iUnit] * normalizedInput[iUnit];
				}
			}
			
			// And then update parameter update speed
			gammaSpeed[iParameter] = (momCoeff * gammaSpeed[iParameter]) - learningRate * gammaGrad;
		}
		else // odd indices => gradient wrt beta
		{
			int iParameter = i /2; // retrieve correct index of parameter

			int iUnit = 0;
			float betaGrad = 0.0F;

			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				iUnit = iExample * inputVolume + iParameter * inputArea; // map beginning
				
				for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
				{
					iUnit += iWithinMap;
					betaGrad += deltaOutput[iUnit];
				}
			}
			
			// And then update parameter update speed
			betaSpeed[iParameter] = (momCoeff * betaSpeed[iParameter]) - learningRate * betaGrad;
		}
	}
}


/* ==================================================================================================================================== */

/* BNFCUPDATESPEEDS()
 * Computes gradients of loss function with respect to learnable parameters gamma and beta
 * in case previous layer is FullyConnected. The gradients are then used to update parameter
 * change speed according to momentum rule and learning rate.
 */

__kernel void 
BNFCUpdateSpeeds(	__global float * gammaSpeed,
					__global float * betaSpeed,
					__global float * deltaOutput,
					__global float * normalizedInput,
					const int nUnits,
					const int miniBatchSize,
					const float momCoeff,
					const float learningRate
				)
{
	// Global work size = number of parameters = 2 * nUnits (should be a bit more efficient)
	const int i = get_global_id(0);
	
	if(i < 2 * nUnits)
	{
		if (i % 2 == 0) // even indices => gradient wrt gamma
		{
			int iParameter = i /2; // retrieve correct index of parameter

			int iUnit = iParameter;
			float gammaGrad = 0.0F;

			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				gammaGrad += deltaOutput[iUnit] * normalizedInput[iUnit];
				
				iUnit += nUnits;
			}
			
			// And then update parameter update speed
			gammaSpeed[iParameter] = (momCoeff * gammaSpeed[iParameter]) - learningRate * gammaGrad;
		}
		else // odd indices => gradient wrt beta
		{
			int iParameter = i /2; // retrieve correct index of parameter

			int iUnit = iParameter;
			float betaGrad = 0.0F;

			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				betaGrad += deltaOutput[iUnit];
				
				iUnit += nUnits;
			}
			
			// And then update parameter update speed
			betaSpeed[iParameter] = (momCoeff * betaSpeed[iParameter]) - learningRate * betaGrad;
		}
	}
}


/* ==================================================================================================================================== */

/* BNUPDATEPARAMETERS()
 * Updates learnable parameters beta and gamma by simply adding the gradient-based update speed.
 * This kernel can be used for both the Conv and the FC case.
 */

__kernel void 
BNUpdateParameters(	__global float * gamma,
					__global float * beta,
					__constant float * gammaSpeed,
					__constant float * betaSpeed,
					const int nGamma // or equally nBeta
						)
{
	// Global work size = inputDepth if Conv : nInputUnits if FC
	int iParameter = get_global_id(0);	
	
	if(iParameter < nGamma)
	{
		gamma[iParameter] += gammaSpeed[iParameter];
		beta[iParameter] += betaSpeed[iParameter];
	}
	
}

/* ==================================================================================================================================== */

/* BNCONVGRADIENTMEANVARIANCE()
 * Computes gradients of loss with respect to mu and sigma^2, needed for backpropagation, in the Conv case.
 */

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
							const int miniBatchSize
							)
{
	// Global work size = number of parameters = 2 * inputDepth (should be a bit more efficient)
	
	const int i = get_global_id(0);							
								
	if (i < 2 * inputDepth)
	{
		const int inputVolume = inputDepth * inputArea;
		
		if (i % 2 == 0) // even indices => compute derivative wrt variance
		{
			int iParameter = i / 2; // retrieve correct index of parameter / feature map
			int iUnit = 0;
			float tmpSumXY = 0.0F;
			
			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				iUnit = iExample * inputVolume + iParameter * inputArea;
			
				for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
				{
					iUnit += iWithinMap;
					tmpSumXY += deltaOutput[iUnit] * normalizedInput[iUnit];
				}
			}
			
			varianceGradient[iParameter] = (-gamma[iParameter] * tmpSumXY) / ( 2*(variance[iParameter] + EPSILON) );
		}
		else // odd indices => compute derivative wrt mean
		{
			int iParameter = i / 2; // retrieve correct index of parameter / feature map
			int iUnit = 0;
			float tmpSumX = 0.0F;
			float tmpSumY = 0.0F;
			float tmpSumXY = 0.0F;
			
			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				iUnit = iExample * inputVolume + iParameter * inputArea;
			
				for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
				{
					iUnit += iWithinMap;
					
					tmpSumX += normalizedInput[iUnit];
					tmpSumY += deltaOutput[iUnit];
					tmpSumXY += deltaOutput[iUnit] * normalizedInput[iUnit];
				}
			}
			
			meanGradient[iParameter] =  (tmpSumX * tmpSumXY / (miniBatchSize * inputArea) - tmpSumY);
			meanGradient[iParameter] *= gamma[iParameter]  / sqrt(variance[iParameter] + EPSILON);
		}
	}		
}


/* ==================================================================================================================================== */

/* BNFCGRADIENTMEANVARIANCE()
 * Computes gradients of loss with respect to mu and sigma^2, needed for backpropagation, in the FC case.
 */

// This is probably going to be very slow :(

__kernel void
BNFCGradientMeanVariance( 	__global float * meanGradient,
							__global float * varianceGradient,
							__global float * deltaOutput,
							__global float * normalizedInput,
							__constant float * gamma,
							__constant float * variance,
							const int nUnits,
							const int miniBatchSize
							)
{
	// Global work size = number of parameters = 2 * nUnits (should be a bit more efficient)
	
	const int i = get_global_id(0);							
								
	if (i < 2 * nUnits)
	{
		if (i % 2 == 0) // even indices => compute derivative wrt variance
		{
			int iParameter = i / 2; // retrieve correct index of parameter / feature map
			int iUnit = iParameter;
			float tmpSumXY = 0.0F;
			
			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				tmpSumXY += deltaOutput[iUnit] * normalizedInput[iUnit];
				
				iUnit += nUnits;
			}
			
			varianceGradient[iParameter] = (-gamma[iParameter] * tmpSumXY) / ( 2*(variance[iParameter] + EPSILON) );
		}
		else // odd indices => compute derivative wrt mean
		{
			int iParameter = i / 2; // retrieve correct index of parameter / feature map
			int iUnit = iParameter;
			float tmpSumX = 0.0F;
			float tmpSumY = 0.0F;
			float tmpSumXY = 0.0F;
			
			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				tmpSumX += normalizedInput[iUnit];
				tmpSumY += deltaOutput[iUnit];
				tmpSumXY += deltaOutput[iUnit] * normalizedInput[iUnit];
				
				iUnit += nUnits;
			}
			
			meanGradient[iParameter] =  (tmpSumX * tmpSumXY / miniBatchSize) - tmpSumY;
			meanGradient[iParameter] *= gamma[iParameter]  / sqrt(variance[iParameter] + EPSILON);
		}
	}		
}


/* ==================================================================================================================================== */

/* BNCONVBACKPROPAGATE()
 * Backpropagates deltaOutput to deltaInput in the Conv case.
 */	

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
					const int miniBatchSize
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


/* ==================================================================================================================================== */

/* BNFCBACKPROPAGATE()
 * Backpropagates deltaOutput to deltaInput in the Conv case.
 */	

__kernel void
BNFCBackPropagate(	__global float * deltaInput,
					__global float * deltaOutput,
					__global float * input,
					__constant float * gamma,
					__constant float * mean,
					__constant float * variance,
					__constant float * meanGradient,
					__constant float * varianceGradient,
					const int nUnits,
					const int miniBatchSize
					)
{
	// Global work size = number of activations = tensor volume * mini-batch size
	const int iUnit = get_global_id(0);
	
	if(iUnit < nUnits * miniBatchSize)
	{
		// Retrieve index of mean / variance / parameters corresponding to this unit
		int iStatistics = iUnit % nUnits;
		
		float tmp = 0.0F;
		
		// See backprop expression...
		tmp += 2 * varianceGradient[iStatistics]* (input[iUnit] - mean[iStatistics]) + meanGradient[iStatistics];
		tmp /= miniBatchSize; 
		tmp += ( (gamma[iStatistics] * deltaOutput[iUnit]) / sqrt(variance[iStatistics] + EPSILON) );
		
		// Write gradient
		deltaInput[iUnit] = tmp;
	}
}

