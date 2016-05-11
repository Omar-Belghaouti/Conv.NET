/*
 * OpenCL kernels used by BatchNormFC layer.
 *	- BNFCComputeMeansVariances;
 *	- BNFCForward;
 *	- BNFCUpdateSpeeds;
 *	- BNUpdateParameters;
 *	- BNFCBackPropagate;
 */
 
#define EPSILON 1.0E-5 // constant small number needed to ensure not to divide by zero when dividing by standard deviation

/* ==================================================================================================================================== */

/* BNFCCOMPUTEMEANSVARIANCES()
 * Computes means and averages over the mini-batch (FC case).
 * These values will then be used to normalize input in kernel BNFCForward().
 * New means and average of the mini-batch are also used to update cumulative averages
 * needed for inference. Note that cumulative averages are automatically reset to zero 
 * at the beginning of a new epoch.
 */

 
// CHECKED (it works, even if it may be VERY slow...)
__kernel void 
BNFCComputeMeansVariances(	__global float * means,
							__global float * variances,
							__global float * cumulativeMeans,
							__global float * cumulativeVariances,
							__global float * input, 
							const int nUnits,
							const int miniBatchSize,
							const int isPreInference,
							const int iCumulativeAverage // index of current miniBatch (needed to cumulate statistics)
				)
{
	// Global work size = number of units in layer
	const int iUnit = get_global_id(0);
	
	if(iUnit < nUnits)
	{
		// First compute  mean
		
		float mean = 0.0F;
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			mean += input[iUnit + iExample*nUnits];
		}
		mean /= miniBatchSize;
		
		// save mean
		means[iUnit] = mean;
		// update cumulative average if pre-inference mode is on
		if (isPreInference > 0)
		{
			cumulativeMeans[iUnit] = (iCumulativeAverage * cumulativeMeans[iUnit] + mean) / (iCumulativeAverage + 1);
		}
		
		// Now compute variance
		
		float centeredInput = 0.0F;
		float variance = 0.0F;
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			centeredInput = input[iUnit + iExample*nUnits] - mean;
			variance += (centeredInput * centeredInput);
		}
		variance /= miniBatchSize;
		
		// Save variance
		variances[iUnit] = variance;
		// update cumulativeVariance if pre-inference mode is on
		if (isPreInference > 0)
		{		
			cumulativeVariances[iUnit] = (iCumulativeAverage * cumulativeVariances[iUnit] + variance) / (iCumulativeAverage + 1);
		}
	}
}

/* ==================================================================================================================================== */

/* BNFCFORWARD()
 * OpenCL kernel for forward pass in BatchNorm layer following a FC layer.
 * Input activations are first normalized using either mean and variance computed
 * over the current mini-batch (if we are training) or the cumulative statistics computed
 * over the last training epoch (if we are doing inference).
 * Then, these normalized values are scaled and shifted using learnable parameters
 * gamma and beta (one pair of parameters per unit).
 */
 
__kernel void 
BNFCForward(__global float * output,
			__global float * normalizedInput,
			__global float * input,
			__constant float * means,		// will be over mini-batch if training, cumulative if inference
			__constant float * variances, 	// same
			__constant float * gamma, 
			__constant float * beta,	
			const int nUnits,
			const int miniBatchSize
			)
{
	// Global work size = number of activations = nUnits * mini-batch size
	const int iActivation = get_global_id(0);
	
	if(iActivation < nUnits * miniBatchSize)
	{
		// Retrieve index of mean / variance corresponding to this unit
		int iUnit = iActivation % nUnits;
		
		// Normalize input, using the pre-calculated mean and variance
		float tmpNormalizedInput = (input[iActivation] - means[iUnit]) * native_rsqrt(variances[iUnit] + EPSILON);
		normalizedInput[iActivation] = tmpNormalizedInput;
		
		// Scale and shift
		output[iActivation] = gamma[iUnit] * tmpNormalizedInput + beta[iUnit];
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
					__global float * deltaGamma, // will be saved and used in BNFCBackPropagate
					__global float * deltaBeta, // will be saved and used in BNFCBackPropagate
					const int nUnits,
					const int miniBatchSize,
					const float momCoeff,
					const float learningRate
				)
{
	// Global work size = number of units
	const int iUnit = get_global_id(0);
	
	if (iUnit < nUnits)
	{
		float gammaGrad = 0.0F;
		float betaGrad = 0.0F;
		int iActivation = iUnit;
		
		for (int iExample = 0; iExample < miniBatchSize; iExample++)
		{
			gammaGrad += (deltaOutput[iActivation] * normalizedInput[iActivation]);
			betaGrad += deltaOutput[iActivation];
			
			iActivation += nUnits;
		}
			
		//EXPERIMENTAL
		gammaGrad /= miniBatchSize;
		betaGrad /= miniBatchSize;
		
		// Save gradient of gamma
		deltaGamma[iUnit] = gammaGrad;
		// And then update parameter update speed
		gammaSpeed[iUnit] = (momCoeff * gammaSpeed[iUnit]) - learningRate * gammaGrad;
	
		
		// Save gradient of beta
		deltaBeta[iUnit] = betaGrad;
		// And then update parameter update speed
		betaSpeed[iUnit] = (momCoeff * betaSpeed[iUnit]) - learningRate * betaGrad;
		
	}
}

/* ==================================================================================================================================== */

/* BNFCUPDATEPARAMETERS()
 * Updates learnable parameters beta and gamma by simply adding the gradient-based update speed.
 * This kernel can be used for both the Conv and the FC case.
 */

__kernel void 
BNFCUpdateParameters(	__global float * gamma,
						__global float * beta,
						__constant float * gammaSpeed,
						__constant float * betaSpeed,
						const int nGamma // or equally nBeta
						)
{
	// Global work size = nInputUnits
	int iParameter = get_global_id(0);	
	
	if(iParameter < nGamma)
	{
		gamma[iParameter] += gammaSpeed[iParameter];
		beta[iParameter] += betaSpeed[iParameter];
	}
	
}


/* ==================================================================================================================================== */

/* BNFCBACKPROPAGATE()
 * Backpropagates deltaOutput to deltaInput in the Conv case.
 */	

__kernel void
BNFCBackPropagate(	__global float * deltaInput,
					__global float * deltaOutput,
					__global float * normalizedInput,
					__constant float * gamma,
					__constant float * variance,
					__constant float * deltaGamma,
					__constant float * deltaBeta,
					const int nUnits,
					const int miniBatchSize
					)
{
	// Global work size = number of activations = tensor volume * mini-batch size
	const int iActivation = get_global_id(0);
	
	if(iActivation < nUnits * miniBatchSize)
	{
		// Retrieve index of unit corresponding to this activation
		int iUnit = iActivation % nUnits;
		
		float tmpDeltaX = 0.0F;
		
		// See backprop expression for how deltaX is computed...
		
		tmpDeltaX = deltaOutput[iActivation] - deltaBeta[iUnit] - (deltaGamma[iUnit] * normalizedInput[iActivation]);
		tmpDeltaX *= ( gamma[iUnit] * native_rsqrt(variance[iUnit] + EPSILON) ); 
		
		
		// Write gradient
		deltaInput[iActivation] = tmpDeltaX;
	}
}

