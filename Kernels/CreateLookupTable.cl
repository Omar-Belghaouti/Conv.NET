__kernel void CreateLookupTable ( 	__write_only __global int* lookupTable,
									const int inputWidth,
									const int outputWidth, // already takes the stride into account
									const int filterSize,
									const int receptiveFieldSize,
									const int stride
								)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
	
	const int nReceptiveFields = outputWidth * outputWidth;
	
	if (i < receptiveFieldSize && j < nReceptiveFields) // check if we are inside the matrix
	{
		const int iReceptiveFieldElement = i;
		const int iReceptiveField = j; 
		
		int iInput = 0; // will be incremented as we "zoom in" step by step
		const int iOutput = iReceptiveFieldElement * nReceptiveFields + iReceptiveField; // destination index
		
		// 0. move to the beginning of the example that we are working on (using j)
		// COMMENTED OUT: not parallelizing over mini-batches now
		//const int iExample = j / nReceptiveFields;
		//const int elementsPerExample = inputDepth * inputWidth * inputWidth;
		//const int iBeginningOfExample = iExample * elementsPerExample;
		//iInput += iBeginningOfExample;
		
		// 1. move to the beginning of channel that we are working on (using i)
		const int iChannel = i / (filterSize * filterSize);
		const int elementsPerChannel = inputWidth*inputWidth;
		const int iBeginningOfChannel = elementsPerChannel * iChannel;
		
		iInput += iBeginningOfChannel;
		
		// 2. now move to the beginning of the receptive field that we are working on (using j)
		// (remember that we are already at the beginning of the correct channel!) 
		const int iOutputRow = j / outputWidth;
		const int iOutputCol = j % outputWidth;
		const int iBeginningOfReceptiveField = iOutputRow * stride * inputWidth + stride * iOutputCol;
		
		iInput += iBeginningOfReceptiveField;
		
		// 3. now move to the correct position within the current receptive field (again, using i)
		// (remember that we are already in the correct channel and receptive field!)
		const int iFilterRow = (i % (filterSize * filterSize)) / filterSize;
		const int iReceptiveFieldCol = i % filterSize;
		const int iWithinReceptiveField = inputWidth * iFilterRow + iReceptiveFieldCol;
		
		iInput += iWithinReceptiveField;
		
		lookupTable[iOutput]= iInput;
	}
	
}