using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;


namespace JaNet
{
    class ConvolutionalLayer : Layer
    {

        #region Fields (private)

        private int filterSize; // F
        private int nFilters; // K
        private int strideLength; // S
        private int zeroPadding; // P
        private int nReceptiveFields;


#if OPENCL_ENABLED
        private Mem paddedInputGPU;
        private Mem paddedOutputGPU; // will not be allocated if not needed

        private int receptiveFieldsMatrixSize;
        private Mem receptiveFieldsMatrixGPU;

        private Mem weightsGPU;
        private Mem biasesGPU;

        private Mem weightsUpdateSpeedGPU;
        private Mem biasesUpdateSpeedGPU;

        // Global and local work-group sizes (for OpenCL kernels) - will be set in SetWorkGroupSizes();
        private IntPtr[] forwardGlobalWorkSizePtr;
        private IntPtr[] forwardLocalWorkSizePtr;
        private IntPtr[] backwardGlobalWorkSizePtr;
        private IntPtr[] backwardLocalWorkSizePtr;
        private IntPtr[] updateGlobalWorkSizePtr;
        private IntPtr[] updateLocalWorkSizePtr;
#else
        private float[] paddedInput; // dimension [inputD * (inputH + 2*padding) * (inutW + 2*padding)]
        private float[] paddedOutput; // dimension [inputD * (inputH + filterSize - 1) * (inutW + filterSize - 1)] <- this makes sure that backprop works

        private float[,] receptiveFieldsMatrix; // dimension [receptiveFieldSize , nReceptiveFields] = [inputDepth*filterSize^2 , outputWidth*outputHeight]
        private float[,] outputMatrix; // dimension [numberOfFilters , outputWidth*outputHeight]

        private float[,] weights; // dimension [nFilters , inputDepth*filterSize^2]
        private float[] biases; // dimension [nFilters , 1]

        private float[,] weightsUpdateSpeed; // dimension [nFilters , inputDepth*filterSize^2]
        private float[] biasesUpdateSpeed; // dimension [nFilters , 1]
#endif

        #endregion


        #region Properties (public)

        #endregion


        #region Setup methods (to be called once)

        /// <summary>
        /// Constructor: specify filter size, number of filters (output depth), stride (only 1 supported at this stage!), zero padding.
        /// </summary>
        /// <param name="FilterSize"></param>
        /// <param name="nOfFilters"></param>
        /// <param name="StrideLength"></param>
        /// <param name="ZeroPadding"></param>
        public ConvolutionalLayer(int FilterSize, int nOfFilters, int StrideLength, int ZeroPadding)
        {
            this.type = "Convolutional";

            this.filterSize = FilterSize;
            this.nFilters = nOfFilters;
            if (StrideLength != 1)
            {
                throw new System.ArgumentException("Stride length > 1 not supported (...yet?)");
            }
            this.strideLength = StrideLength;
            this.zeroPadding = ZeroPadding;
        }



        public override void ConnectTo(Layer PreviousLayer)
        {
            // Setup input
            base.ConnectTo(PreviousLayer);

            if (PreviousLayer.OutputHeight != PreviousLayer.OutputWidth)
                throw new ArgumentException("ConvolutionalLayer currently only supports square input (spatially).");

            this.inputWidth = PreviousLayer.OutputWidth;
            this.inputHeight = PreviousLayer.OutputHeight;
            this.inputDepth = PreviousLayer.OutputDepth;

            // Setup output
            double tmp = (double)(inputWidth - filterSize + 2 * zeroPadding) / (double)strideLength + 1; // then check if this number is int
            if (Math.Abs(tmp % 1) > Global.EPSILON) 
                throw new System.ArgumentException("Input width, filter size, zero padding and stride length do not fit well. Use different values");
            this.outputWidth = (int)tmp;
            this.outputHeight = (int)tmp;

            this.outputDepth = nFilters;

            this.outputNeurons = new Neurons(nFilters * outputWidth * outputHeight);

            // Padded I/O
            int paddedInputSize = inputDepth * (inputHeight + 2 * zeroPadding) * (inputWidth + 2 * zeroPadding);
            int outputPadding = (filterSize - 1) / 2;
            int paddedOutputSize = outputDepth * (outputHeight + 2 * outputPadding) * (outputWidth + 2 * outputPadding);

#if OPENCL_ENABLED
            this.paddedInputGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context, 
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)paddedInputSize,
                                                        out OpenCLSpace.ClError);

            if (paddedOutputSize == paddedInputSize)
                this.paddedOutputGPU = paddedInputGPU; // will forever point here, no new allocations needed
            else
            {
                this.paddedOutputGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)paddedOutputSize,
                                                            out OpenCLSpace.ClError);
            }
#else
            this.paddedInput = new float[paddedInputSize];
            if (paddedOutputSize == paddedInputSize)
                this.paddedOutput = paddedInput; // will forever point here, no new allocations needed
            else
                this.paddedOutput = new float[paddedOutputSize];
#endif

            // Receptive fields matrix
#if OPENCL_ENABLED

            this.receptiveFieldsMatrixSize = sizeof(float) * (inputDepth * filterSize ^ 2 * outputWidth * outputHeight);
            this.receptiveFieldsMatrixGPU = (Mem)Cl.CreateBuffer(   OpenCLSpace.Context, 
                                                                    MemFlags.ReadWrite,
                                                                    (IntPtr)receptiveFieldsMatrixSize,
                                                                    out OpenCLSpace.ClError);
#else
            this.receptiveFieldsMatrix  = new float[inputDepth * filterSize * filterSize, outputWidth * outputHeight];
            this.outputMatrix = new float[nFilters, outputWidth * outputHeight];
#endif
            // Work group sizes
#if OPENCL_ENABLED
            SetWorkGroupSizes();
#endif
        }

#if OPENCL_ENABLED
        private void SetWorkGroupSizes()
        {
            // The code below is _UGLY_, but unfortunately I could not come up with a neater implementation...

            // Work group sizes will be set as follows:
            //      global work size = total number of processes needed
            //      local work size = largest divisor of global work size <= maxWorkGroupSize of device in context
            // (this is probably suboptimal, but improvements are most likely negligible compared to improvements elsewhere, e.g. in the kernels code)

            // FeedForward __________________________________________________________________________________________
            this.forwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(nFilters), (IntPtr)(outputHeight * outputWidth) };
            int[] tmp = new int[] { nFilters, outputHeight * outputWidth };
            // make each local work group dimension <= corresponding max work item size (depends on device)
            while (tmp[0] > OpenCLSpace.MaxWorkItemSizes[0] && tmp[0] % 2 == 0)
                tmp[0] /= 2;
            while (tmp[1] > OpenCLSpace.MaxWorkItemSizes[1] && tmp[1] % 2 == 0)
                tmp[1] /= 2;
            // make entire local work group size (i.e. product of dimensions) <= of max work group size (depends on device)
            while (tmp[0] * tmp[1] > OpenCLSpace.MaxWorkGroupSize && tmp[1] % 2 == 0)
            {
                tmp[1] /= 2;
            }
            while (tmp[0] * tmp[1] > OpenCLSpace.MaxWorkGroupSize && tmp[0] % 2 == 0)
            {
                tmp[0] /= 2;
                if (tmp[0] == 1)
                {
                    throw new System.ArgumentException("I can't set a suitable local work group size! :(");
                }
            }
            this.forwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(tmp[0]), (IntPtr)(tmp[1]) };

            // BackPropagate __________________________________________________________________________________________
            this.backwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(InputNeurons.NumberOfUnits) };
            int tmpBwLocalWorkSize = InputNeurons.NumberOfUnits;
            while (tmpBwLocalWorkSize > OpenCLSpace.MaxWorkGroupSize || tmpBwLocalWorkSize > OpenCLSpace.MaxWorkItemSizes[0])
                tmpBwLocalWorkSize /= 2;
            this.backwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)(tmpBwLocalWorkSize) };

            // UpdateParameters
            this.updateGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(OutputNeurons.NumberOfUnits), (IntPtr)(InputNeurons.NumberOfUnits) };
            int[] tmpUpdLocalWorkSize = new int[] { OutputNeurons.NumberOfUnits, InputNeurons.NumberOfUnits };
            // make each local work group dimension <= corresponding max work item size (depends on device)
            while (tmpUpdLocalWorkSize[0] > OpenCLSpace.MaxWorkItemSizes[0] && tmpUpdLocalWorkSize[0] % 2 == 0)
                tmpUpdLocalWorkSize[0] /= 2;
            while (tmpUpdLocalWorkSize[1] > OpenCLSpace.MaxWorkItemSizes[1] && tmpUpdLocalWorkSize[1] % 2 == 0)
                tmpUpdLocalWorkSize[1] /= 2;
            // make entire local work group size (i.e. product of dimensions) <= of max work group size (depends on device)
            while (tmpUpdLocalWorkSize[0] * tmpUpdLocalWorkSize[1] > OpenCLSpace.MaxWorkGroupSize && tmpUpdLocalWorkSize[1] % 2 == 0)
            {
                tmpUpdLocalWorkSize[1] /= 2;
            }
            while (tmpUpdLocalWorkSize[0] * tmpUpdLocalWorkSize[1] > OpenCLSpace.MaxWorkGroupSize && tmpUpdLocalWorkSize[0] % 2 == 0)
            {
                tmpUpdLocalWorkSize[0] /= 2;
                if (tmpUpdLocalWorkSize[0] == 1)
                {
                    throw new System.InvalidOperationException("I can't set a suitable local work group size! :(");
                }
            }
            this.updateLocalWorkSizePtr = new IntPtr[] { (IntPtr)(tmpUpdLocalWorkSize[0]), (IntPtr)(tmpUpdLocalWorkSize[1]) };

        }
#endif


        public override void InitializeParameters()
        {
            // Initialize weigths as normally distributed numbers with mean 0 and std equals to 1/sqrt(numberOfInputUnits)
            // Initialize biases as small positive numbers, e.g. 0.01

            this.weights = new float[nFilters, inputDepth * filterSize * filterSize];
            this.biases = new float[nFilters];

            double weightsStdDev = Math.Sqrt(2.0 / this.input.NumberOfUnits);
            double uniformRand1;
            double uniformRand2;
            double tmp;

            for (int iRow = 0; iRow < weights.GetLength(0); iRow++)
            {
                for (int iCol = 0; iCol < weights.GetLength(1); iCol++)
                {
                    uniformRand1 = Global.rng.NextDouble();
                    uniformRand2 = Global.rng.NextDouble();
                    // Use a Box-Muller transform to get a random normal(0,1)
                    tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);
                    tmp = weightsStdDev * tmp; // rescale using stdDev

                    weights[iRow, iCol] = (float)tmp;
                }

                //uniformRand1 = rng.NextDouble();
                //uniformRand2 = rng.NextDouble();
                // Use a Box-Muller transform to get a random normal(0,1)
                //tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);

                biases[iRow] = 0.01F;//(float)tmp;

            }

            // Also initialize updates speeds to zero (for momentum)
            this.weightsUpdateSpeed = new float[nFilters, inputDepth * filterSize * filterSize];
            this.biasesUpdateSpeed = new float[nFilters];
        }

        #endregion


        #region Training methods

        public override void FeedForward()
        {
#if OPENCL_ENABLED
            //TODO:
            // implement forward pass with OpenCL
#else
            this.inputAsMatrix = UnrollInput(input.GetHost());
            this.outputAsMatrix = Utils.MatrixMultiply(weights, inputAsMatrix);
            this.output.SetHost(OutputMatrixToVector(outputAsMatrix));
#endif
        }

        public override void BackPropagate()
        {

        }


        public override void UpdateParameters(double learningRate, double momentumCoefficient)
        {
        }

        #endregion


        #region Private methods

        /// <summary>
        /// Reshape input vector into a matrix of receptive fields, so that convolution can be implemented as matrix multiplication (fast!).
        /// This method is likely to be incredibly SLOW and will soon be ported to OpenCL.
        /// </summary>
        /// <param name="inputVector"></param>
        /// <returns></returns>
        [Obsolete("This method is slow, use the OpenCL kernel instead.")]
        private float[,] UnrollInput(float[] inputVector)
        {
            int nRows = inputDepth * filterSize * filterSize;
            int nCols = outputWidth * outputWidth;
            float[,] unrolledInput = new float[nRows, nCols];

            // Unfortunately there is no way of writing this so that it is readable!
            for (int i = 0; i < nRows; i++)
            {
                int iChannelBeginning = inputWidth * inputHeight * (i / (filterSize * filterSize));

                int iAux1 = (i % filterSize) + inputWidth * ((i % (filterSize * filterSize)) / filterSize);

                for (int j = 0; j < nCols; j++)
                {
                    int iAux2 = (j % outputWidth) + inputWidth * (j / outputWidth);

                    unrolledInput[i, j] = inputVector[iChannelBeginning + iAux1 + iAux2];
                }
            }

            return unrolledInput;
        }

        /// <summary>
        /// Reshape input vector to a matrix so that convolution can be implemented as matrix multiplication (fast!).
        /// This method is likely to be incredibly SLOW and will soon be ported to OpenCL.
        /// </summary>
        /// <param name="outputMatrix"></param>
        /// <returns></returns>
        [Obsolete("Replace this method with an OpenCL kernel!")]
        private float[] OutputMatrixToVector(float[,] outputMatrix)
        {


            int nRows = nFilters;
            int nCols = outputWidth * outputHeight;

            float[] reshapedOutput = new float[nFilters * outputWidth * outputHeight];

            for (int i = 0; i < nRows; i++)
            {
                for (int j = 0; j < nCols; j++)
                {
                    reshapedOutput[i * nCols + j] = outputMatrix[i, j];
                }
            }

            return reshapedOutput;
        }

        /// <summary>
        /// Pad input vector with zeros.
        /// </summary>
        /// <param name="array"></param>
        /// <param name="padding"></param>
        /// <param name="depth"></param>
        /// <param name="height"></param>
        /// <param name="width"></param>
        /// <returns></returns>
        [Obsolete("Replace this method with an OpenCL kernel!")]
        static float[] PadWithZeros(float[] array, int padding, int depth, int height, int width)
        {
            int area = height * width;
            int volume = depth * height * width;
            int zerosPerSlice = 2 * padding * (height + width + 2 * padding);
            float[] paddedArray = new float[array.Length + depth * zerosPerSlice];

            // auxiliary variables
            int iRow, iSlice, iNew;

            for (int k = 0; k < array.Length; k++)
            {
                iRow = (int)((k % area) / width);
                iSlice = (int)((k % volume) / area);

                iNew = k + padding + padding * (2 * padding + width) + 2 * padding * iRow + zerosPerSlice * iSlice;

                paddedArray[iNew] = array[k];
            }

            return paddedArray;
        }


        #endregion

    
    }
}
