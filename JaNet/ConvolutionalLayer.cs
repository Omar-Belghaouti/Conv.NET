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

        private int receptiveFieldSize; // i.e. [outputDepth * filterSize^2]
        private int nReceptiveFields; // i.e. output depth

        // aux
        private int unpaddedVolume;
        private int paddedVolume;

#if OPENCL_ENABLED
        
        private Mem paddedInputBatchGPU;
        private Mem paddingLookupTableGPU;
        private Mem recFieldsLookupTableGPU;

        private Mem weightsGPU;
        private Mem biasesGPU;

        private Mem weightsSpeedGPU;
        private Mem biasesSpeedGPU;

        // Global and local work-group sizes (for OpenCL kernels) - will be set in SetWorkGroupSizes();

        private IntPtr[] paddingGlobalWorkSizePtr;
        private IntPtr[] paddingLocalWorkSizePtr;

        private IntPtr[] forwardGlobalWorkSizePtr;
        private IntPtr[] forwardLocalWorkSizePtr;

        private IntPtr[] backwardGlobalWorkSizePtr;
        private IntPtr[] backwardLocalWorkSizePtr;

        private IntPtr[] updateParametersGlobalWorkSizePtr;
        private IntPtr[] updateParametersLocalWorkSizePtr;

#else
        private List<double[]> paddedInput; // dimension [inputD * (inputH + 2*padding) * (inutW + 2*padding)]
        private int[,] lookupTable; // dimension [receptiveFieldSize , nReceptiveFields] = [inputDepth*filterSize^2 , outputWidth*outputHeight]
        

        private double[,] weights; // dimension [nFilters , inputDepth*filterSize^2]
        private double[] biases; // dimension [nFilters , 1]

        private double[,] weightsGradients;
        private double[] biasesGradients;

        private double[,] weightsUpdateSpeed; // dimension [nFilters , inputDepth*filterSize^2]
        private double[] biasesUpdateSpeed; // dimension [nFilters , 1]
#endif

        #endregion


        #region Properties (public)

        
#if GRADIENT_CHECK
        // accessors for gradient check

        public override double[,] Weights
        {
            get { return weights; }
            set { this.weights = value; }
        }

        public override double[] Biases
        {
            get { return biases; }
            set { this.biases = value; }
        }

        public override double[,] WeightsGradients 
        {
            get { return weightsGradients; }
        }

        public override double[] BiasesGradients
        {
            get { return biasesGradients; }
        }
#endif
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

            if (FilterSize % 2 != 1)
                throw new ArgumentException("Only odd filter size is supported.");
            this.filterSize = FilterSize;
            this.nFilters = nOfFilters;
            this.strideLength = StrideLength;
            this.zeroPadding = ZeroPadding;
        }


        public override void SetupOutput()
        {
            // Check that input is spatially square
            if (inputHeight != inputWidth)
                throw new ArgumentException("ConvolutionalLayer currently only supports square input (spatially).");

            // Setup output __________________________________________________________________________________________

            // Check if parameters fit
            double tmp = (double)(inputWidth - filterSize + 2 * zeroPadding) / (double)strideLength + 1;
            if (Math.Abs(tmp % 1) > Global.EPSILON) 
                throw new System.ArgumentException("Input size, filter size, padding and stride length do not fit well (non-integer output size).");
            this.outputWidth = (int)tmp;
            this.outputHeight = (int)tmp;
            this.outputDepth = nFilters;

            this.nReceptiveFields = outputHeight * outputWidth;
            this.receptiveFieldSize = inputDepth * filterSize * filterSize;

            this.nOutputUnits = outputDepth * outputWidth * outputHeight;
            this.outputNeurons = new Neurons(nOutputUnits);

            this.unpaddedVolume = inputDepth * inputHeight * inputWidth;
            this.paddedVolume = inputDepth * (inputHeight + 2 * zeroPadding) * (inputWidth + 2 * zeroPadding);


#if OPENCL_ENABLED
            // Initialize auxiliary structures  __________________________________________________________________________________________ 

            if (zeroPadding > 0)
            {
                // 1) Padded input buffer (and wipe it, just in case)

                this.paddedInputBatchGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)(sizeof(float) * paddedVolume * inputNeurons.MiniBatchSize),
                                                                out OpenCLSpace.ClError);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.CreateBuffer paddedInputBatchGPU");
                OpenCLSpace.WipeBuffer(paddedInputBatchGPU, paddedVolume * inputNeurons.MiniBatchSize, typeof(float));

                // 2) Zero-padding lookup table buffer (and wipe it, just in case)

                this.paddingLookupTableGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                    MemFlags.ReadWrite,
                                                                    (IntPtr)(sizeof(int) * inputDepth * inputHeight * inputWidth),
                                                                    out OpenCLSpace.ClError);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.CreateBuffer paddingLookupTableGPU");
                OpenCLSpace.WipeBuffer(paddingLookupTableGPU, inputDepth * inputHeight * inputWidth, typeof(int));
                // Note that this is the same for every input example, no need to create miniBatchSize copies of it!
            }

            // 3) Receptive fields lookup table buffer (and wipe it, just in case)

            this.recFieldsLookupTableGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)(sizeof(int) * receptiveFieldSize * nReceptiveFields),
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ConnectTo(): Cl.CreateBuffer receptiveFieldsLookupTableGPU");
            OpenCLSpace.WipeBuffer(recFieldsLookupTableGPU, receptiveFieldSize * nReceptiveFields, typeof(int));
            // Note that this is the same for every input example, no need to create miniBatchSize copies of it!


            // Create lookup tables once and for all  ____________________________________________________________________________________

            // 1) Padding lookup table

            if (zeroPadding > 0)
            {
                // Set kernel arguments
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.CreatePaddingLookupTable, 0, paddingLookupTableGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.CreatePaddingLookupTable, 1, (IntPtr)sizeof(int), inputWidth);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.CreatePaddingLookupTable, 2, (IntPtr)sizeof(int), inputDepth);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.CreatePaddingLookupTable, 3, (IntPtr)sizeof(int), zeroPadding);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg CreatePaddingLookupTable");

                // These work group sizes have a limited scope and therefore they are not class fields
                IntPtr[] tmp1DLocalWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.OPTIMAL_GROUP_SIZE };
                int smallestMultiple = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE *
                    Math.Ceiling((double)(inputDepth * inputHeight * inputWidth) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
                IntPtr[] tmp1DGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple };

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                                OpenCLSpace.CreatePaddingLookupTable,
                                                                1,
                                                                null,
                                                                tmp1DGlobalWorkSizePtr,
                                                                tmp1DLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel CreatePaddingLookupTable");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
            }

            // 2) Receptive fields lookup table

            // Set kernel arguments
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.CreateRecFieldsLookupTable, 0, recFieldsLookupTableGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.CreateRecFieldsLookupTable, 1, (IntPtr)sizeof(int), inputWidth + 2 * zeroPadding);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.CreateRecFieldsLookupTable, 2, (IntPtr)sizeof(int), outputWidth);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.CreateRecFieldsLookupTable, 3, (IntPtr)sizeof(int), filterSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.CreateRecFieldsLookupTable, 4, (IntPtr)sizeof(int), receptiveFieldSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.CreateRecFieldsLookupTable, 5, (IntPtr)sizeof(int), strideLength);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg CreateRecFieldsLookupTable");

            // These work sizes have a limited scope and therefore they are not class fields
            int baseToOptimalFactor = OpenCLSpace.OPTIMAL_GROUP_SIZE / OpenCLSpace.BASE_GROUP_SIZE;
            IntPtr[] tmp2DLocalWorkSizePtr = new IntPtr[] { (IntPtr)baseToOptimalFactor, (IntPtr)OpenCLSpace.BASE_GROUP_SIZE };
            int smallestMultipleReceptiveFieldSize = (int)(baseToOptimalFactor * Math.Ceiling((double)receptiveFieldSize / (double)baseToOptimalFactor));
            int smallestMultipleNReceptiveFields = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)nReceptiveFields / (double)OpenCLSpace.BASE_GROUP_SIZE));
            IntPtr[] tmp2DGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultipleReceptiveFieldSize, (IntPtr)smallestMultipleNReceptiveFields };

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.CreateRecFieldsLookupTable,
                                                            2,
                                                            null,
                                                            tmp2DGlobalWorkSizePtr,
                                                            tmp2DLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel CreateRecFieldsLookupTable");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            // Cpu code

            this.paddedInput = new List<double[]>();
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
                paddedInput.Add(new double[paddedInputSize]);

            this.lookupTable = new int[receptiveFieldSize, nReceptiveFields];
            this.lookupTable = CreateLookupTableCPU();
#endif
        }


        public override void SetWorkGroups()
        {
#if OPENCL_ENABLED
            // Work group sizes will be set as follows:
            //      global work size = smallest multiple of OPTIMAL_GROUP_SIZE larger than 
            //                         the total number of processes needed (for efficiency).
            //      local work size = as close as possible to OPTIMAL_GROUP_SIZE (making sure 
            //                        that global worksize is a multiple of this)
            // OPTIMAL_GROUP_SIZE is a small multiple of BASE_GROUP_SIZE, which in turn is a 
            //                    constant multiple of 2, platform-dependent, e.g. 32 (Nvidia 
            //                    WARP) or 64 (AMD WAVEFRONT).

            // Zero padding / unpadding (1D workspace) _________________________________________________________________________________

            if (zeroPadding > 0)
            {
                // Local
                int localWorkSize = OpenCLSpace.OPTIMAL_GROUP_SIZE;
                this.paddingLocalWorkSizePtr = new IntPtr[] { (IntPtr)localWorkSize };

                // Global
                int smallestMultiple = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE *
                    Math.Ceiling((double)(inputDepth * inputHeight * inputWidth * inputNeurons.MiniBatchSize) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
                this.paddingGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple };
            }

            // Forward kernel (2D workspace) ___________________________________________________________________________________________

            // Local
            int optimalToBaseFactor = OpenCLSpace.OPTIMAL_GROUP_SIZE / OpenCLSpace.BASE_GROUP_SIZE;
            this.forwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)optimalToBaseFactor, (IntPtr)OpenCLSpace.BASE_GROUP_SIZE }; // product is optimal

            // Global
            int nRowsOutput = inputNeurons.MiniBatchSize * nFilters;
            int smallestMultipleOutputDepthBatch = (int)(optimalToBaseFactor * Math.Ceiling((double)(nRowsOutput) / (double)optimalToBaseFactor));
            int smallestMultipleNReceptiveFields = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)nReceptiveFields / (double)OpenCLSpace.BASE_GROUP_SIZE ) );
            this.forwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultipleOutputDepthBatch, (IntPtr)smallestMultipleNReceptiveFields };


            // Backward kernel (2D workspace) __________________________________________________________________________________________

            // Local
            this.backwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)optimalToBaseFactor, (IntPtr)OpenCLSpace.BASE_GROUP_SIZE };

            // Global
            int nRowsDeltaX = inputNeurons.MiniBatchSize * receptiveFieldSize;
            int smallestMultipleRowsDeltaX = (int)(optimalToBaseFactor * Math.Ceiling((double)(nRowsDeltaX) / (double)optimalToBaseFactor));
            this.backwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultipleRowsDeltaX, (IntPtr)smallestMultipleNReceptiveFields };
            

            // Update parameters kernel (2D workspace) ___________________________________________________________________________________
            
            // Local
            this.updateParametersLocalWorkSizePtr = new IntPtr[] { (IntPtr)optimalToBaseFactor, (IntPtr)OpenCLSpace.BASE_GROUP_SIZE };

            // Global
            int smallestMultipleNFilters = (int)(optimalToBaseFactor * Math.Ceiling( (double)(nFilters) / (double)optimalToBaseFactor) );
            int smallestMultipleReceptiveFieldSize = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling( (double)(receptiveFieldSize) / (double)OpenCLSpace.BASE_GROUP_SIZE ) );
            this.updateParametersGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultipleNFilters, (IntPtr)smallestMultipleReceptiveFieldSize };
#endif
        }



        public override void InitializeParameters()
        {
            // Initialize weigths as normally distributed numbers with mean 0 and std equals to 1/sqrt(numberOfInputUnits)
            // Initialize biases as small positive numbers, e.g. 0.01

#if OPENCL_ENABLED
            float[,] initWeights = new float[nFilters, receptiveFieldSize];
            float[] initBiases = new float[nFilters];
#else
            this.weights = new double[nFilters, receptiveFieldSize];
            this.biases = new double[nFilters];
#endif

            double weightsStdDev = Math.Sqrt(2.0 / this.inputNeurons.NumberOfUnits);
            double uniformRand1;
            double uniformRand2;
            double tmp;

            for (int iRow = 0; iRow < nFilters; iRow++)
            {
                for (int iCol = 0; iCol < receptiveFieldSize; iCol++)
                {
                    uniformRand1 = Global.rng.NextDouble();
                    uniformRand2 = Global.rng.NextDouble();
                    // Use a Box-Muller transform to get a random normal(0,1)
                    tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);
                    tmp = weightsStdDev * tmp; // rescale using stdDev
#if OPENCL_ENABLED
                    initWeights[iRow, iCol] = (float)tmp;
#else
                    weights[iRow, iCol] = tmp;
#endif
                }
#if OPENCL_ENABLED
                initBiases[iRow] = 0.01f;
#else
                biases[iRow] = 0.01;
#endif
            }


#if OPENCL_ENABLED
            // Create weights and biases buffers and copy initial values

            int weightBufferSize = sizeof(float) * (nFilters * receptiveFieldSize);
            int biasesBufferSize = sizeof(float) * nFilters;

            this.weightsGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                    (IntPtr)weightBufferSize,
                                                    initWeights,
                                                    out OpenCLSpace.ClError);
            this.biasesGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                    (IntPtr)biasesBufferSize,
                                                    initBiases,
                                                    out OpenCLSpace.ClError);

            // Also create weightsSpeed and biasesSpeed buffers, without copying anything.
            // This USUALLY means they're initialized to zeros...

            this.weightsSpeedGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)weightBufferSize,
                                                        out OpenCLSpace.ClError);
            this.biasesSpeedGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)biasesBufferSize,
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            // ...but better make extra sure and enforce this.
            OpenCLSpace.WipeBuffer(weightsSpeedGPU, nFilters * receptiveFieldSize, typeof(float));
            OpenCLSpace.WipeBuffer(biasesSpeedGPU, nFilters, typeof(float));

#else

            this.weightsUpdateSpeed = new double[nFilters, receptiveFieldSize]; // zeors
            this.biasesUpdateSpeed = new double[nFilters]; // zeros

            this.weightsGradients = new double[nFilters, receptiveFieldSize];
            this.biasesGradients = new double[nFilters];
#endif


        }

        #endregion


        #region Training methods
        

        public override void FeedForward()
        {
#if OPENCL_ENABLED

            // 1. Zero-pad input tensor (if necessary) _________________________________________________________

            if (zeroPadding > 0)
            {
                // Set kernel arguments
                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.ZeroPadBatch, 0, paddedInputBatchGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPadBatch, 1, inputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPadBatch, 2, paddingLookupTableGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPadBatch, 3, (IntPtr)sizeof(int), unpaddedVolume);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPadBatch, 4, (IntPtr)sizeof(int), paddedVolume);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPadBatch, 5, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg ZeroPadBatch");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.ZeroPadBatch,
                                                                1,
                                                                null,
                                                                paddingGlobalWorkSizePtr,
                                                                paddingLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel ZeroPadBatch");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
            }

            // 2. Convolve input and filter bank _________________________________________________________

            // Set kernel arguments
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ConvForwardBatch, 0, outputNeurons.ActivationsGPU);
            if (zeroPadding > 0)
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForwardBatch, 1, paddedInputBatchGPU);
            else
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForwardBatch, 1, inputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForwardBatch, 2, recFieldsLookupTableGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForwardBatch, 3, weightsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForwardBatch, 4, biasesGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForwardBatch, 5, (IntPtr)sizeof(int), nFilters);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForwardBatch, 6, (IntPtr)sizeof(int), receptiveFieldSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForwardBatch, 7, (IntPtr)sizeof(int), nReceptiveFields);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForwardBatch, 8, (IntPtr)sizeof(int), paddedVolume);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForwardBatch, 9, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg ConvForwardBatch");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.ConvForwardBatch,
                                                            2,
                                                            null,
                                                            forwardGlobalWorkSizePtr,
                                                            forwardLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel ConvForwardBatch");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                if (zeroPadding > 0)
                    paddedInput[m] = ZeroPadCPU(inputNeurons.GetHost()[m], zeroPadding, inputDepth, inputHeight, inputWidth);
                else
                    paddedInput[m] = inputNeurons.GetHost()[m];

                outputNeurons.SetHost(m, ConvForwardCPU(paddedInput[m]));
            }
#endif

        }

        public override void BackPropagate()
        {
            //Console.WriteLine("Checkpoint C");

#if OPENCL_ENABLED
            // 1. Wipe out input buffers where we are going to write gradients 
            // (this is important because gradients wrt different locations in receptive field have to be summed!)

            if (zeroPadding > 0)
                OpenCLSpace.WipeBuffer(paddedInputBatchGPU, paddedVolume * inputNeurons.MiniBatchSize, typeof(float));
            else
                OpenCLSpace.WipeBuffer(inputNeurons.DeltaGPU, unpaddedVolume * inputNeurons.MiniBatchSize, typeof(float));
            
            // 2. Convolution backpropagation as matrix multiplication (see ConvBackPropagate kernel)
                
            // Set kernel arguments
            if (zeroPadding > 0)
            {
                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.ConvBackPropagateBatch, 0, paddedInputBatchGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagateBatch, 1, (IntPtr)sizeof(int), paddedVolume);
            }
            else
            {
                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.ConvBackPropagateBatch, 0, inputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagateBatch, 1, (IntPtr)sizeof(int), unpaddedVolume);
            }
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagateBatch, 2, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagateBatch, 3, weightsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagateBatch, 4, recFieldsLookupTableGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagateBatch, 5, (IntPtr)sizeof(int), nFilters);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagateBatch, 6, (IntPtr)sizeof(int), receptiveFieldSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagateBatch, 7, (IntPtr)sizeof(int), nReceptiveFields);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagateBatch, 8, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg ConvBackPropagateBatch");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.ConvBackPropagateBatch,
                                                            2,
                                                            null,
                                                            backwardGlobalWorkSizePtr,
                                                            backwardLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel ConvBackPropagateBatch");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            // 3. (if needed) Unpad paddedInputBatchGPU to get deltaX

            if (zeroPadding > 0)
            {
                // Set kernel arguments
                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.ZeroUnpadBatch, 0, inputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpadBatch, 1, paddedInputBatchGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpadBatch, 2, paddingLookupTableGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpadBatch, 3, (IntPtr)sizeof(int), unpaddedVolume);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpadBatch, 4, (IntPtr)sizeof(int), paddedVolume);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpadBatch, 5, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg ZeroUnpadBatch");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.ZeroUnpadBatch,
                                                                1,
                                                                null,
                                                                paddingGlobalWorkSizePtr,
                                                                paddingLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel ZeroUnpadBatch");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

                //Console.WriteLine("Checkpoint D");

            }
#else
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                // 1. Wipe out deltaX buffer (will be cumulated!)

                //Array.Clear(paddedInput[m], 0, paddedInputSize); // no longer needed
                
                // 2. Backpropagate error

                paddedInput[m] = ConvBackwardCPU(outputNeurons.DeltaHost[m]);

                // 3. Unpad (if necessary)

                if (zeroPadding > 0)
                    inputNeurons.DeltaHost[m] = ZeroUnpadCPU(paddedInput[m]);
                else
                    inputNeurons.DeltaHost[m] = paddedInput[m];
            } // end of loop over mini-batch
#endif
        }


        public override void UpdateSpeeds(double learningRate, double momentumCoefficient)
        {
            //Console.WriteLine("Checkpoint A");
#if OPENCL_ENABLED
            // Set kernel arguments
            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 0, weightsSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 1, biasesSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 2, outputNeurons.DeltaGPU);
            if (zeroPadding > 0)
            {
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 3, paddedInputBatchGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 4, (IntPtr)sizeof(int), paddedVolume);
            }
            else
            {
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 3, inputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 4, (IntPtr)sizeof(int), unpaddedVolume);
            }
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 5, recFieldsLookupTableGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 6, (IntPtr)sizeof(int), nFilters);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 7, (IntPtr)sizeof(int), receptiveFieldSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 8, (IntPtr)sizeof(int), nReceptiveFields);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 9, (IntPtr)sizeof(float), (float)momentumCoefficient);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 10, (IntPtr)sizeof(float), (float)learningRate);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeedsBatch, 11, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.UpdateSpeeds(): Cl.SetKernelArg ConvUpdateSpeedsBatch");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.ConvUpdateSpeedsBatch,
                                                            2,
                                                            null,
                                                            updateParametersGlobalWorkSizePtr,
                                                            updateParametersLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.UpdateSpeeds(): Cl.EnqueueNDRangeKernel ConvUpdateSpeedsBatch");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            //Console.WriteLine("Checkpoint B");
#else

            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                ConvGradientsCPU(ref weightsGradients, ref biasesGradients, outputNeurons.DeltaHost[m], paddedInput[m]);

                //double gradientNorm = 0;

                for (int iFilter = 0; iFilter < nFilters; iFilter++)
                {
                    for (int iElement = 0; iElement < receptiveFieldSize; iElement++)
                    {
                        weightsUpdateSpeed[iFilter, iElement] *= momentumCoefficient;
                        weightsUpdateSpeed[iFilter, iElement] -= (learningRate * weightsGradients[iFilter, iElement]);
                        //gradientNorm += Math.Pow(weightsGradients[iFilter, iElement], 2);
                    }

                    // update biases
                    biasesUpdateSpeed[iFilter] *= momentumCoefficient;
                    biasesUpdateSpeed[iFilter] -= (learningRate * biasesGradients[iFilter]);
                }

                //gradientNorm = Math.Sqrt(gradientNorm);

                //Console.WriteLine("Layer {0}\n\tGradient norm: {1}", this.ID, gradientNorm);
                //if (gradientNorm < Global.EPSILON)
                    //Console.WriteLine("BUSTED");
                    //System.Diagnostics.Debugger.Launch();
            }
#endif

        }

        public override void UpdateParameters()
        {
            //Console.WriteLine("Checkpoint E");

#if OPENCL_ENABLED
            // Set kernel arguments
            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.ConvUpdateParameters, 0, weightsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateParameters, 1, biasesGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateParameters, 2, weightsSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateParameters, 3, biasesSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateParameters, 4, (IntPtr)sizeof(int), nFilters);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateParameters, 5, (IntPtr)sizeof(int), receptiveFieldSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.UpdateParameters(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.ConvUpdateParameters,
                                                            2,
                                                            null,
                                                            updateParametersGlobalWorkSizePtr,
                                                            updateParametersLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.UpdateParameters(): Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            //Console.WriteLine("Checkpoint F");
#else
            //double weightNorm = 0;
            //double updateNorm = 0;

            for (int iFilter = 0; iFilter < nFilters; iFilter++)
            {
                // weights update

                for (int iElement = 0; iElement < receptiveFieldSize; iElement++)
                {
                    //weightNorm += Math.Pow(weights[iFilter, iElement], 2);
                    //updateNorm += Math.Pow(weightsUpdateSpeed[iFilter, iElement], 2);

                    weights[iFilter, iElement] += weightsUpdateSpeed[iFilter, iElement];
                    
                }   

                // update biases
                biases[iFilter] += biasesUpdateSpeed[iFilter];
            }

            //weightNorm = Math.Sqrt(weightNorm);
            //updateNorm = Math.Sqrt(updateNorm);

            //Console.WriteLine("\tWeight norm: {0}\n\tSpeed norm: {1}\n\tRatio: {2}", weightNorm, updateNorm, updateNorm / weightNorm );
            //Console.WriteLine("Speed/weight ratio: {0}", updateNorm / weightNorm);
            //Console.ReadKey();
#endif
        }

        #endregion


        #region CPU private methods

#if !OPENCL_ENABLED
        private static double[] ZeroPadCPU(double[] array, int padding, int depth, int height, int width)
        {
            int area = height * width;
            int volume = depth * height * width;
            int nZerosTopRows = padding * (2 * padding + width);
            int zerosPerSlice = 2 * padding * (height + width + 2 * padding);

            double[] paddedArray = new double[depth * (height + 2 * padding) * (width + 2 * padding)];

            // auxiliary variables
            int iRow, iSlice, iOutput;

            for (int iInput = 0; iInput < array.Length; iInput++)
            {
                iSlice = (iInput % volume) / area; // find index of channel within an input volume
                iRow = (iInput % area) / width; // find index of row within an input channel
                iOutput = zerosPerSlice * iSlice + nZerosTopRows + padding * (2 * iRow + 1) + iInput;

                paddedArray[iOutput] = array[iInput];
            }

            return paddedArray;
        }


        private double[] ZeroUnpadCPU(double[] paddedArray)
        {
            int area = inputHeight * inputWidth;
            int volume = inputDepth * inputHeight * inputWidth;
            int nZerosTopRows = zeroPadding * (2 * zeroPadding + inputWidth);
            int zerosPerSlice = 2 * zeroPadding * (inputHeight + inputWidth + 2 * zeroPadding);

            double[] unpaddedArray = new double[volume];

            // auxiliary variables
            int iRow, iSlice, iPadded;

            for (int iUnpadded = 0; iUnpadded < volume; iUnpadded++)
            {
                iSlice = (iUnpadded % volume) / area; // find index of channel within an input volume
                iRow = (iUnpadded % area) / inputWidth; // find index of row within an input channel
                iPadded = zerosPerSlice * iSlice + nZerosTopRows + zeroPadding * (2 * iRow + 1) + iUnpadded;

                unpaddedArray[iUnpadded] = paddedArray[iPadded];
            }

            return unpaddedArray;
        }

        private int[,] CreateLookupTableCPU()
        {
            int nReceptiveFields = outputWidth * outputWidth;
            int[,] lookupTable = new int[receptiveFieldSize, nReceptiveFields];

            for (int i = 0; i < receptiveFieldSize; i++)
            {
                int iReceptiveFieldElement = i;

                // 1. move to the beginning of channel that we are working on (using i)
                int iChannel = i / (filterSize * filterSize);
                int elementsPerChannel = inputWidth * inputWidth;
                int iBeginningOfChannel = elementsPerChannel * iChannel;

                // 3. now move to the correct position within the current receptive field (again, using i)
                // (remember that we are already in the correct channel and receptive field!)
                int iFilterRow = (i % (filterSize * filterSize)) / filterSize;
                int iReceptiveFieldCol = i % filterSize;
                int iWithinReceptiveField = inputWidth * iFilterRow + iReceptiveFieldCol;

                for (int j = 0; j < nReceptiveFields; j++)
                {
                    int iReceptiveField = j;

                    int iInput = 0; // will be incremented as we "zoom in" step by step

                    // 0. move to the beginning of the example that we are working on (using j)
                    // COMMENTED OUT: not parallelizing over mini-batches now
                    //const int iExample = j / nReceptiveFields;
                    //const int elementsPerExample = inputDepth * inputWidth * inputWidth;
                    //const int iBeginningOfExample = iExample * elementsPerExample;
                    //iInput += iBeginningOfExample;

                    iInput += iBeginningOfChannel;

                    iInput += iWithinReceptiveField;

                    // 2. now move to the beginning of the receptive field that we are working on (using j)
                    // (remember that we are already at the beginning of the correct channel!) 
                    int iOutputRow = j / outputWidth;
                    int iOutputCol = j % outputWidth;
                    int iBeginningOfReceptiveField = iOutputRow * strideLength * inputWidth + strideLength * iOutputCol;

                    iInput += iBeginningOfReceptiveField;

                    lookupTable[iReceptiveFieldElement, iReceptiveField] = iInput;
                }

            }

            return lookupTable;
        }


        private double[] ConvForwardCPU(double[] input)
        {
            double[] output = new double[nFilters * nReceptiveFields];

            for (int iFilter = 0; iFilter < nFilters; iFilter++)
            {
                for (int iReceptiveField = 0; iReceptiveField < nReceptiveFields; iReceptiveField++)
                {
                    double sum = 0.0F;

                    for (int iElement = 0; iElement < receptiveFieldSize; iElement++)
                    {
                        // Get filter element needed 
                        double filterElement = weights[iFilter, iElement];

                        // Get receptive field element needed, reading it from 
                        // inputPadded indexed using the receptive field lookup table
                        double receptiveFieldElement = input[lookupTable[iElement, iReceptiveField]];

                        // Multiply & cumulate in sum
                        sum += filterElement * receptiveFieldElement;
                    }

                    // Add bias
                    sum += biases[iFilter];

                    // Finally, write output buffer
                    output[iFilter * nReceptiveFields + iReceptiveField] = sum;
                }
            }

            return output;
        }

        private double[] ConvBackwardCPU(double[] deltaY)
        {
            double[] deltaX = new double[paddedInputSize];

            for (int iElement = 0; iElement < receptiveFieldSize; iElement++)
            {
                for (int iReceptiveField = 0; iReceptiveField < nReceptiveFields; iReceptiveField++)
                {
                    double tmpDeltaX = 0.0F;

                    for (int iFilter = 0; iFilter < nFilters; iFilter++)
                    {
                        // Get filter element from transpose of wSpeeds
                        double filterElement = weights[iFilter, iElement];

                        // Get error signal corresponding to this filter and this receptiveField
                        double deltaElement = deltaY[iFilter * nReceptiveFields + iReceptiveField];

                        // Multiply & cumulate in gradW
                        tmpDeltaX += filterElement * deltaElement;
                    }

                    // Now cumulate this in correct place of deltaX (using lookup table)
                    int inputLocation = lookupTable[iElement, iReceptiveField];
                    deltaX[inputLocation] += tmpDeltaX;
                }
            }

            return deltaX;
        }

        
        private void ConvGradientsCPU(ref double[,] weightsGradients, ref double[] biasesGradients, double[] deltaY, double[] input)
        {

            for (int iFilter = 0; iFilter < nFilters; iFilter++)
            {
                for (int iElement = 0; iElement < receptiveFieldSize; iElement++)
                {
                    double tmpGradW = 0.0F;
                    double tmpGradB = 0.0F;

                    for(int iReceptiveField = 0; iReceptiveField < nReceptiveFields; iReceptiveField++)
		            {
			            // Get error signal corresponding to this filter and this receptiveField
                        double deltaElement = deltaY[iFilter * nReceptiveFields + iReceptiveField];
			
			            // Get input value needed, reading it from transpose(input) indexed using the receptive field lookup table
                        double inputElement = input[lookupTable[iElement, iReceptiveField]];
			
			            // Multiply & cumulate in gradW
                        tmpGradW += deltaElement * inputElement;
			
			            // Once per filter, cumulate error signals in gradB
			            if (iElement == 0)
			            {
                            tmpGradB += deltaElement;
			            }
		            }

                    weightsGradients[iFilter, iElement] = tmpGradW;

                    if (iElement == 0)
                    {
                        biasesGradients[iFilter] = tmpGradB;
                    }

                }
            }


        }
        
#endif

        #endregion
        
    
    }
}
