//TODO: Generalise to miniBatchSize > 1

__kernel void 
CheckClassification(__global int* assignedClass,
					__global float* networkOutput,
					//__constant float* trueLabel,
					const int nClasses
                )
{
	uint global_id = get_global_id(0);	
	
	if (global_id == 0) {
		int iMaxScore = 0;
		float maxScore = networkOutput[0];
		for (uint i = 1; i < nClasses; i++) {
			if (networkOutput[i] > maxScore) {
				iMaxScore = i;
				maxScore = networkOutput[i];
			}
				
		}
		
		*assignedClass = iMaxScore;
		//if (iMaxScore == *trueLabel)
		//	*error = 0;
		//else
		//	*error = 1;
			
	}
}