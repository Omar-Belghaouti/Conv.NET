/*
 * OpenCL kernels used by BatchNormConv layer.
 *	- BNConvComputeMeansVariances;
 *	- BNConvForward;
 *	- BNConvUpdateSpeeds;
 *	- BNUpdateParameters;
 *	- BNConvBackPropagate;
 */
 
#define EPSILON 1.0E-5 // constant small number needed to ensure not to divide by zero when dividing by standard deviation



/* ==================================================================================================================================== */

/* BNCONVCOMPUTEMEANSVARIANCES()
 * Computes means and averages over spatial dimension AND mini-batch (Conv case).
 * These values will then be used to normalize input in kernel BatchNormConvForward().
 * New means and average of the mini-batch are also used to update cumulative averages
 * needed for inference. Note that cumulative averages are automatically reset to zero 
 * at the beginning of a new epoch, using index iCumulativeAverage.
 */

 
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
							const int isPreInference,
							const int iCumulativeAverage // index of current sample of mu and sigma^2 (should be between 0 and dataSetSize/miniBatchSize)
				)
{
	// Global work size = number of feature maps of previous layer = nFilters
	const int iFeatureMap = get_global_id(0);
	
	if(iFeatureMap < inputDepth)
	{
		int iMapBeginning = iFeatureMap * inputArea; // beginning of this map in example 0
		
		float mean = 0.0f;
		float variance = 0.0f;
		
		// Compute mean
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
			{
				mean += input[iMapBeginning+iWithinMap];
			}
			
			iMapBeginning += inputVolume;
		}
		mean /= (miniBatchSize * inputArea);
		
		// save mean 
		means[iFeatureMap] = mean;
		// update cumulative average if pre-inference mode is on
		if (isPreInference > 0)
		{
			cumulativeMeans[iFeatureMap] = (iCumulativeAverage * cumulativeMeans[iFeatureMap] + mean) / (iCumulativeAverage + 1);
		}
		
		// Then compute variance
		
		float centeredInput = 0.0f;
		
		iMapBeginning = iFeatureMap * inputArea;
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
			{
				centeredInput = input[iMapBeginning+iWithinMap] - mean;
				variance += centeredInput * centeredInput;
			}
		}
		variance /= (miniBatchSize * inputArea - 1);
		
		// Save variance
		variances[iFeatureMap] = variance;		
		// update cumulative variance if pre-inference mode is on
		if (isPreInference > 0)
		{
			cumulativeVariances[iFeatureMap] = (iCumulativeAverage * cumulativeVariances[iFeatureMap] + variance) / (iCumulativeAverage + 1);
		}
		
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
				__constant float * gamma, 
				__constant float * beta,	
				const int inputArea,
				const int inputVolume,
				const int miniBatchSize
				)
{
	// Global work size = number of activations = tensor volume * mini-batch size
	const int iActivation = get_global_id(0);
	
	if(iActivation < inputVolume * miniBatchSize)
	{
		// Retrieve index of feature map that this input activation belongs to
		int iFeatureMap = (iActivation % inputVolume) / inputArea;
		
		// Normalize input, using the pre-calculated mean and variance
		float privateNormalizedInput = (input[iActivation] - means[iFeatureMap]) * native_rsqrt(variances[iFeatureMap] + EPSILON);
		// Save it (needed in backprop)
		normalizedInput[iActivation] = privateNormalizedInput;
		
		// Scale and shift
		output[iActivation] = gamma[iFeatureMap] * privateNormalizedInput + beta[iFeatureMap];
	}
}


/* ==================================================================================================================================== */

/* BNCONVPARAMETERGRADIENTSBATCH()
 * First step in computation of the gradients of loss function with respect to learnable parameters gamma and beta:
 * Gradient is first cumulated over the mini-batch. In the next step it will be cumulated over the feature map.
 */
 
 // STEP 1
 
__kernel void 
BNConvParameterGradientsBatch(	__global float * deltaGammaBatch,
								__global float * deltaBetaBatch,
								__global float * deltaOutput,
								__global float * normalizedInput,
								const int inputVolume,
								const int miniBatchSize
				)
{
	// Global work size = number of units = inputVolume = nInputUnits
	const int iUnit = get_global_id(0);
	
	if(iUnit < inputVolume)
	{
		float tmpGammaGrad = 0.0F;
		float tmpBetaGrad = 0.0F;
		
		int iActivation = iUnit;
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			tmpGammaGrad += deltaOutput[iActivation] * normalizedInput[iActivation];
			tmpBetaGrad += deltaOutput[iActivation];
			
			iActivation += inputVolume;
		}
		
		deltaGammaBatch[iUnit] = tmpGammaGrad;
		deltaBetaBatch[iUnit] = tmpBetaGrad;
	}
}



/* ==================================================================================================================================== */

/* BNCONVUPDATESPEEDS()
 * Computes gradients of loss function with respect to learnable parameters gamma and beta, cumulating deltaGammaStep1 and deltaBetaStep1
 * over each feature map. This should be considerably faster than doing the sum in one step with few work items doing a lot of work.
 * The gradients are then saved (needed in BackPropagate) and used to update parameter change speed.
 */
 
 // STEP 2
 
__kernel void 
BNConvUpdateSpeeds(	__global float * gammaSpeed,
					__global float * betaSpeed,
					__global float * deltaGamma,
					__global float * deltaBeta,
					__global float * deltaGammaBatch,
					__global float * deltaBetaBatch,
					const int inputDepth,
					const int inputArea,
					const float momCoeff,
					const float learningRate,
					const int miniBatchSize
				)
{
	// Global work size = number of feature maps = inputDepth
	const int iFeatureMap = get_global_id(0);
	
	if(iFeatureMap < inputDepth)
	{
		int iMapBeginning = iFeatureMap * inputArea;
		int iTmpGrad = 0;
		
		float gammaGrad = 0.0F;
		float betaGrad = 0.0F;
		
		for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
		{
			iTmpGrad = iMapBeginning + iWithinMap;
			
			gammaGrad += deltaGammaBatch[iTmpGrad];
			betaGrad += deltaBetaBatch[iTmpGrad];
		}
		
		// EXPERIMENTAL
		gammaGrad /= miniBatchSize;
		betaGrad /= miniBatchSize;
		
		// Save gradients
		deltaGamma[iFeatureMap] = gammaGrad;
		deltaBeta[iFeatureMap] = betaGrad;
		
		// And then update parameter update speeds
		gammaSpeed[iFeatureMap] = (momCoeff * gammaSpeed[iFeatureMap]) - learningRate * gammaGrad;
		betaSpeed[iFeatureMap] = (momCoeff * betaSpeed[iFeatureMap]) - learningRate * betaGrad;
		
	}
}





/* ==================================================================================================================================== */

/* BNCONVUPDATESPEEDS()
 * Computes gradients of loss function with respect to learnable parameters gamma and beta.
 * The gradients are then saved and used to update parameter change speed.
 
 
 // ONESTEP (BACKUP... PROBABLY BUGGY!)
__kernel void 
BNConvUpdateSpeeds(	__global float * gammaSpeed,
					__global float * betaSpeed,
					__global float * deltaGamma,
					__global float * deltaBeta,
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
		if ( (i & 1) == 0) // even indices => gradient wrt gamma (this is the same as computing the modulo 2, but faster)
		{
			
			int iFeatureMap = i /2; // retrieve correct index of parameter / feature map
			int iMapBeginning = iFeatureMap * inputArea; // beginning of this map in example 0
			int iActivation = 0;
			
			float gammaGrad = 0.0F;

			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
				{
					iActivation = iMapBeginning + iWithinMap;
					//gammaGrad += deltaOutput[iActivation] * normalizedInput[iActivation];
					gammaGrad = fma(deltaOutput[iActivation], normalizedInput[iActivation], gammaGrad);
				}
				iMapBeginning += inputVolume;
			}
			// Save gradient
			deltaGamma[iFeatureMap] = gammaGrad;
			// And then update parameter update speed
			gammaSpeed[iFeatureMap] = (momCoeff * gammaSpeed[iFeatureMap]) - learningRate * gammaGrad;
		}
		else // odd indices => gradient wrt beta
		{
			
			int iFeatureMap = i / 2; // retrieve correct index of parameter / feature map
			int iMapBeginning = iFeatureMap * inputArea; // beginning of this map in example 0
			
			float betaGrad = 0.0F;

			for (int iExample = 0; iExample < miniBatchSize; iExample++)
			{
				for (int iWithinMap = 0; iWithinMap < inputArea; iWithinMap++)
				{
					betaGrad += deltaOutput[iMapBeginning + iWithinMap];
				}
				iMapBeginning += inputVolume;
			}
			// Save gradient
			deltaBeta[iFeatureMap] = betaGrad;
			// And then update parameter update speed
			betaSpeed[iFeatureMap] = (momCoeff * betaSpeed[iFeatureMap]) - learningRate * betaGrad;
			
		}
		
	}
}


/* ==================================================================================================================================== */

/* BNCONVUPDATEPARAMETERS()
 * Updates learnable parameters beta and gamma by simply adding the gradient-based update speed.
 * This kernel can be used for both the Conv and the FC case.
 */

__kernel void 
BNConvUpdateParameters(	__global float * gamma,
						__global float * beta,
						__constant float * gammaSpeed,
						__constant float * betaSpeed,
						const int nFeatureMaps
						)
{
	// Global work size = inputDepth
	int iParameter = get_global_id(0);	
	
	if(iParameter < nFeatureMaps)
	{
		gamma[iParameter] += gammaSpeed[iParameter];
		beta[iParameter] += betaSpeed[iParameter];
	}
}


/* ==================================================================================================================================== */

/* BNCONVBACKPROPAGATE()
 * Backpropagates deltaOutput to deltaInput in the Conv case.
 */	

__kernel void
BNConvBackPropagate(__global float * deltaInput,
					__global float * deltaOutput,
					__global float * normalizedInput,
					__constant float * gamma,
					__constant float * variance,
					__constant float * deltaGamma,
					__constant float * deltaBeta,
					const int inputArea,
					const int inputVolume,
					const int miniBatchSize
					)
{
	// Global work size = number of activations = tensor volume * mini-batch size
	const int iActivation = get_global_id(0);
	
	if(iActivation < inputVolume * miniBatchSize)
	{
		// Retrieve index of feature map that this input activation belongs to
		int iFeatureMap = (iActivation % inputVolume) / inputArea;
		int iMapBeginning = iFeatureMap * inputArea;
		int iWithinMap = iActivation % inputArea;
		
		float tmpDeltaX = 0.0F;
		
		// See backprop expression for how deltaX is computed...
		
		tmpDeltaX = deltaOutput[iActivation] - deltaBeta[iMapBeginning+iWithinMap] / inputArea - deltaGamma[iFeatureMap] * normalizedInput[iActivation] / inputArea;
		tmpDeltaX *= ( gamma[iFeatureMap] * native_rsqrt(variance[iFeatureMap] + EPSILON) ); 
		
		// Write gradient
		deltaInput[iActivation] = tmpDeltaX;
	}
}
