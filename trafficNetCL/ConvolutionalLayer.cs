using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace JaNet
{
    class ConvolutionalLayer : Layer
    {

        #region Fields (private)

        private int filterSize; // F
        private int nFilters; // K
        private int strideLength; // S
        private int zeroPadding; // P

        private float[,] inputAsMatrix; // dimension [inputDepth*filterSize^2 , outputWidth*outputHeight]
        private float[,] outputAsMatrix; // dimension [nFilters , outputWidth*outputHeight]

        private float[,] weights; // dimension [nFilters , inputDepth*filterSize^2]
        private float[] biases; // dimension [nFilters , 1]

        private float[,] weightsUpdateSpeed; // dimension [nFilters , outputWidth*outputHeight]
        private float[] biasesUpdateSpeed; // dimension [nFilters , 1]

        #endregion


        #region Properties (public)

        #endregion


        #region Setup methods (to be called once)

        // TO-DO: implement constructor
        public ConvolutionalLayer(int FilterSize, int nOfFilters, int StrideLength, int ZeroPadding)
        {
            //Console.WriteLine("Adding a convolutional layer with K = {0} filters of size F = {1}, stride length S = {2} and zero padding P = {3}...",
            //    filterSize, nOfFilters, strideLength, zeroPadding);

            this.type = "Convolutional";

            this.filterSize = FilterSize;
            this.nFilters = nOfFilters;
            this.strideLength = StrideLength;
            this.zeroPadding = ZeroPadding;
        }

        public override void ConnectTo(Layer PreviousLayer)
        {
            // Setup input
            base.ConnectTo(PreviousLayer);
            if ((PreviousLayer.Type != "Convolutional") & (PreviousLayer.Type != "Pooling"))
                throw new System.InvalidOperationException("Attaching convolutional layer to a non-convolutional or non-pooling layer.");
            
            this.inputWidth = PreviousLayer.OutputWidth;
            this.inputHeight = PreviousLayer.OutputHeight;
            this.inputDepth = PreviousLayer.OutputDepth;

            // Setup output
            double tmp = (double)(inputWidth - filterSize + 2 * zeroPadding) / (double)strideLength + 1;
            if (Math.Abs(tmp % 1) > GlobalVar.EPSILON)
                throw new System.ArgumentException("Input width, filter size, zero padding and stride length do not fit well. Check the values!");
            this.outputWidth = (int) tmp;
            tmp = (double)(inputHeight - filterSize + 2 * zeroPadding) / (double)strideLength + 1;
            if (Math.Abs(tmp % 1) > GlobalVar.EPSILON)
                throw new System.ArgumentException("Input height, filter size, zero padding and stride length do not fit well. Check the values!");
            this.outputHeight = (int)tmp;
            this.outputDepth = nFilters;
            this.output = new Neurons( nFilters * outputWidth * outputHeight);
            
            // Setup I/O matrices
            this.inputAsMatrix = new float[inputDepth * filterSize * filterSize, outputWidth * outputHeight ];
            this.outputAsMatrix = new float[nFilters, outputWidth * outputHeight];
        }

        

        /// <summary>
        /// Method to setup the first layer of the network.
        /// </summary>
        public override void SetAsFirstLayer(int InputWidth, int InputHeight, int InputDepth)
        {
            // Setup input
            this.inputWidth = InputWidth;
            this.inputHeight = InputHeight;
            this.inputDepth = InputDepth;
            this.input = new Neurons(InputDepth * InputWidth * InputHeight);

            // Setup output
            double tmp = (double)(inputWidth - filterSize + 2 * zeroPadding) / (double)strideLength + 1;
            Console.WriteLine("Output width = {0}", tmp);
            if (Math.Abs(tmp % 1) > GlobalVar.EPSILON)
                throw new System.ArgumentException("Input width, filter size, zero padding and stride length do not fit well. Check the values!");
            this.outputWidth = (int) tmp;
            tmp = (double)(inputHeight - filterSize + 2 * zeroPadding) / (double)strideLength + 1;
            if (Math.Abs(tmp % 1) > GlobalVar.EPSILON)
                throw new System.ArgumentException("Input height, filter size, zero padding and stride length do not fit well. Check the values!");
            this.outputHeight = (int) tmp;
            this.outputDepth = nFilters;
            this.output = new Neurons(nFilters * outputWidth * outputHeight);

            // Setup I/O matrices
            this.inputAsMatrix = new float[inputDepth * filterSize * filterSize, outputWidth * outputHeight];
            this.outputAsMatrix = new float[nFilters, outputWidth * outputHeight];
        }

        public override void InitializeParameters()
        {
            // Initialize weigths as normally distributed numbers with mean 0 and std equals to 1/sqrt(numberOfInputUnits)
            // Initialize biases as small positive numbers, e.g. 0.01

            this.weights = new float[nFilters, inputDepth * filterSize * filterSize];
            this.biases = new float[nFilters];

            Random rng = new Random();
            double weightsStdDev = Math.Sqrt(2.0 / this.input.NumberOfUnits);
            double uniformRand1;
            double uniformRand2;
            double tmp;

            for (int iRow = 0; iRow < weights.GetLength(0); iRow++)
            {
                for (int iCol = 0; iCol < weights.GetLength(1); iCol++)
                {
                    uniformRand1 = rng.NextDouble();
                    uniformRand2 = rng.NextDouble();
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

        public override void ForwardOneCPU()
        {
            this.inputAsMatrix = UnrollInput(input.Get());
            this.outputAsMatrix = Utils.MatrixMultiply(weights, inputAsMatrix);
            this.output.Set(OutputMatrixToVector(outputAsMatrix));
            // Probably implementing all of this as a single OpenCL kernel would be a good idea
        }

        public override void ForwardBatchCPU()
        {
        }

        public override void ForwardGPU()
        {
        }

        public override void BackPropOneCPU()
        {

        }

        public override void BackPropBatchCPU()
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
