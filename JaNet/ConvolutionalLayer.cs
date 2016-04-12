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

        private int paddedInputSize;
        private int receptiveFieldsLookupTableSize;

        private int inputArea;
        private int inputVolume;
        private int nZerosTopRows;
        private int nZerosPerSlice;

#if OPENCL_ENABLED
        
        private List<Mem> paddedInputGPU;
        private Mem receptiveFieldsLookupTableGPU;

        private Mem weightsGPU;
        private Mem biasesGPU;

        private Mem weightsSpeedGPU;
        private Mem biasesSpeedGPU;

        // Global and local work-group sizes (for OpenCL kernels) - will be set in SetWorkGroupSizes();

        private IntPtr[] paddingGlobalWorkSizePtr;
        private IntPtr[] paddingLocalWorkSizePtr;

        private IntPtr[] im2colGlobalWorkSizePtr;
        private IntPtr[] im2colLocalWorkSizePtr;

        private IntPtr[] forwardGlobalWorkSizePtr;
        private IntPtr[] forwardLocalWorkSizePtr;

        private IntPtr[] backwardGlobalWorkSizePtr;
        private IntPtr[] backwardLocalWorkSizePtr;

        private IntPtr[] weightsUpdateSpeedGlobalWorkSizePtr;
        private IntPtr[] weightsUpdateSpeedLocalWorkSizePtr;

        private IntPtr[] biasesUpdateSpeedGlobalWorkSizePtr;
        private IntPtr[] biasesUpdateSpeedLocalWorkSizePtr;

        private IntPtr[] updateParametersGlobalWorkSizePtr;
        private IntPtr[] updateParametersLocalWorkSizePtr;

#else
        private float[] paddedInput; // dimension [inputD * (inputH + 2*padding) * (inutW + 2*padding)]
        private float[,] receptiveFieldsLookupTable; // dimension [receptiveFieldSize , nReceptiveFields] = [inputDepth*filterSize^2 , outputWidth*outputHeight]
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

            if (FilterSize % 2 != 1)
                throw new ArgumentException("Only odd filter size is supported.");
            this.filterSize = FilterSize;
            this.nFilters = nOfFilters;
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
            this.nReceptiveFields = outputHeight * outputWidth;

            this.outputDepth = nFilters;
            this.receptiveFieldSize = inputDepth * filterSize * filterSize;

            this.outputNeurons = new Neurons(outputDepth * outputWidth * outputHeight);

            this.inputArea = inputWidth * inputHeight;
            this.inputVolume = inputWidth * inputHeight * inputDepth;
            this.nZerosTopRows = zeroPadding * (2 * zeroPadding + inputWidth);
            this.nZerosPerSlice = 2 * zeroPadding * (inputWidth + inputHeight + 2 * zeroPadding);

            this.paddedInputSize = inputDepth * (inputHeight + 2 * zeroPadding) * (inputWidth + 2 * zeroPadding);
            this.receptiveFieldsLookupTableSize = receptiveFieldSize * nReceptiveFields;

#if OPENCL_ENABLED

            // Padded input buffer

            this.paddedInputGPU = new List<Mem>();
            this.paddedInputGPU.Add( (Mem)Cl.CreateBuffer(  OpenCLSpace.Context, 
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * paddedInputSize),
                                                            out OpenCLSpace.ClError) );
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ConnectTo(): Cl.CreateBuffer paddedInputGPU");

            // Receptive fields lookup table buffer

            this.receptiveFieldsLookupTableGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                                        MemFlags.ReadWrite,
                                                                        (IntPtr)(sizeof(int) * receptiveFieldsLookupTableSize),
                                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ConnectTo(): Cl.CreateBuffer receptiveFieldsLookupTableGPU");

            

            // (no need for output matrix: will be written directly to OuptutNeurons.ActivationsGPU
#else
            // Cpu code

            this.paddedInput = new float[paddedInputSize];
            this.receptiveFieldsLookupTable = new float[receptiveFieldSize, nReceptiveFields];
            this.outputMatrix = new float[nFilters, nReceptiveFields];
#endif

           
#if OPENCL_ENABLED
            // Set all work group sizes
            SetWorkGroupSizes();

            // We're ready to create the lookup table once and for all

            // Set kernel arguments
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.Im2colLookupTable, 0, receptiveFieldsLookupTableGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.Im2colLookupTable, 1, (IntPtr)sizeof(int), inputWidth);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.Im2colLookupTable, 2, (IntPtr)sizeof(int), outputWidth);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.Im2colLookupTable, 3, (IntPtr)sizeof(int), filterSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.Im2colLookupTable, 4, (IntPtr)sizeof(int), receptiveFieldSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.Im2colLookupTable, 5, (IntPtr)sizeof(int), strideLength);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ConnectTo(): Cl.SetKernelArg Im2colLookupTable");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.Im2colLookupTable,
                                                            2,
                                                            null,
                                                            im2colGlobalWorkSizePtr,
                                                            im2colLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ConvolutionalLayer.ConnectTo() Im2colLookupTable() Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

#endif
        }

#if OPENCL_ENABLED
        private void SetWorkGroupSizes()
        {
            // Work group sizes will be set as follows:
            //      global work size = smallest multiple of BASE_GROUP_SIZE larger than the total number of processes needed (for efficiency)
            //      local work size = BASE_GROUP_SIZE or small multiples of it (making sure that global worksize is a multiple of this)
            // BASE_GROUP_SIZE is a constant multiple of 2. Suggested values: 32 (Nvidia) or 64 (AMD).


            // TODO: also make sure that each local work group size is lesser than KERNEL_WORK_GROUP_SIZE

            // Zero padding / unpadding (1D workspace) _________________________________________________________________________________
            int inputSize = inputDepth * inputWidth * inputHeight;
            int smallestMultipleInputSize = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(inputSize) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.paddingGlobalWorkSizePtr = new IntPtr[] {(IntPtr)smallestMultipleInputSize};
            int localWorkSize = OpenCLSpace.BASE_GROUP_SIZE;
            int maxKernelWorkGroupSize = Cl.GetKernelWorkGroupInfo( OpenCLSpace.ZeroPad, 
                                                                    OpenCLSpace.Device, 
                                                                    KernelWorkGroupInfo.WorkGroupSize, 
                                                                    out OpenCLSpace.ClError).CastTo<int>();
            while (localWorkSize <= OpenCLSpace.MaxWorkGroupSize && localWorkSize <= maxKernelWorkGroupSize)
            {
                int tmpLocalWorkSize = 2*localWorkSize;
                if (smallestMultipleInputSize % tmpLocalWorkSize == 0) // if global divides local
                    localWorkSize = tmpLocalWorkSize;
                else
                    break;
            }
            this.paddingLocalWorkSizePtr = new IntPtr[] { (IntPtr)localWorkSize };
 

            // Receptive field lookup table (2D workspace) _____________________________________________________________________________
            IntPtr smallestMultipleReceptiveFieldSize = (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(receptiveFieldSize) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            IntPtr smallestMultipleNReceptiveFields = (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(nReceptiveFields) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.im2colGlobalWorkSizePtr = new IntPtr[] { smallestMultipleReceptiveFieldSize, smallestMultipleNReceptiveFields };
            this.im2colLocalWorkSizePtr = new IntPtr[] { (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE/4), (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE/2) };

            // Forward kernel (2D workspace) ___________________________________________________________________________________________
            IntPtr smallestMultipleOutputDepth = (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(outputDepth) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.forwardGlobalWorkSizePtr = new IntPtr[] { smallestMultipleOutputDepth, smallestMultipleNReceptiveFields };
            this.forwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE / 4), (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE / 2) };

            // Backward kernel (2D workspace) __________________________________________________________________________________________
            this.backwardGlobalWorkSizePtr = new IntPtr[] { smallestMultipleReceptiveFieldSize, smallestMultipleNReceptiveFields };
            this.backwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE / 4), (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE / 2) };

            // Weights gradient kernel (2D workspace) ___________________________________________________________________________________
            //this.weightsUpdateSpeedGlobalWorkSizePtr = new IntPtr[] { smallestMultipleOutputDepth, smallestMultipleReceptiveFieldSize };
            //this.weightsUpdateSpeedLocalWorkSizePtr = new IntPtr[] { (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE / 4), (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE / 2) };

            // Biases gradient kernel (1D workspace) ___________________________________________________________________________________
            //this.biasesUpdateSpeedGlobalWorkSizePtr = new IntPtr[] { smallestMultipleOutputDepth };
            //this.biasesUpdateSpeedLocalWorkSizePtr = new IntPtr[] { (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE * 2)};

            // Update parameters kernel (2D workspace) ___________________________________________________________________________________
            this.updateParametersGlobalWorkSizePtr = new IntPtr[] { smallestMultipleOutputDepth, smallestMultipleReceptiveFieldSize };
            this.updateParametersLocalWorkSizePtr = new IntPtr[] { (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE / 4), (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE / 2) };
        }
#endif


        public override void InitializeParameters()
        {
            // Initialize weigths as normally distributed numbers with mean 0 and std equals to 1/sqrt(numberOfInputUnits)
            // Initialize biases as small positive numbers, e.g. 0.01

            float[,] initWeights = new float[nFilters, receptiveFieldSize];
            float[] initBiases = new float[nFilters];

            float[,] initWeightsUpdateSpeed = new float[nFilters, receptiveFieldSize]; // zeros
            float[] initBiasesUpdateSpeed = new float[nFilters]; // zeros

            double weightsStdDev = Math.Sqrt(2.0 / this.inputNeurons.NumberOfUnits);
            double uniformRand1;
            double uniformRand2;
            double tmp;

            for (int iRow = 0; iRow < initWeights.GetLength(0); iRow++)
            {
                for (int iCol = 0; iCol < initWeights.GetLength(1); iCol++)
                {
                    uniformRand1 = Global.rng.NextDouble();
                    uniformRand2 = Global.rng.NextDouble();
                    // Use a Box-Muller transform to get a random normal(0,1)
                    tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);
                    tmp = weightsStdDev * tmp; // rescale using stdDev

                    initWeights[iRow, iCol] = (float)tmp;
                }

                initBiases[iRow] = 0.01F;
            }


#if OPENCL_ENABLED
            // initialize parameter buffers and write these random initial values to them

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

            // Also initialize update speeds (to zero)

            this.weightsSpeedGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                                (IntPtr)weightBufferSize,
                                                                initWeightsUpdateSpeed,
                                                                out OpenCLSpace.ClError);
            this.biasesSpeedGPU = (Mem)Cl.CreateBuffer(   OpenCLSpace.Context,
                                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                                (IntPtr)biasesBufferSize,
                                                                initBiasesUpdateSpeed,
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
#else

            this.weights = initWeights;
            this.biases = initBiases;

            this.weightsUpdateSpeed = initWeightsUpdateSpeed;
            this.biasesUpdateSpeed = initBiasesUpdateSpeed;
#endif
        }

        #endregion


        #region Training methods
        

        public override void FeedForward()
        {
#if OPENCL_ENABLED

            // inelegant workaround to add memory buffers to List<Mem> paddedInputGPU in case miniBatchSize > 1
            // TODO: find a more elegant solution for this
            while (inputNeurons.MiniBatchSize > paddedInputGPU.Count)
            {
                this.paddedInputGPU.Add((Mem)Cl.CreateBuffer(   OpenCLSpace.Context,
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)(sizeof(float) * paddedInputSize),
                                                                out OpenCLSpace.ClError));
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ConnectTo(): Cl.CreateBuffer paddedInputGPU");
            }

            // Forward method begins here

            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {

                // 1. Zero-pad input tensor _________________________________________________________

                // Set kernel arguments
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ZeroPad, 0, inputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPad, 1, paddedInputGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPad, 2, (IntPtr)sizeof(int), inputWidth);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPad, 3, (IntPtr)sizeof(int), inputArea);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPad, 4, (IntPtr)sizeof(int), inputVolume);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPad, 5, (IntPtr)sizeof(int), zeroPadding);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPad, 6, (IntPtr)sizeof(int), nZerosTopRows);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroPad, 7, (IntPtr)sizeof(int), nZerosPerSlice);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.FeedForward(): Cl.SetKernelArg ZeroPad");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.ZeroPad,
                                                                1,
                                                                null,
                                                                paddingGlobalWorkSizePtr,
                                                                paddingLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.FeedForward(): Cl.EnqueueNDRangeKernel ZeroPad");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                // 2. Convolve input and filters _________________________________________________________

                // Set kernel arguments
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ConvForward, 0, outputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForward, 1, paddedInputGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForward, 2, receptiveFieldsLookupTableGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForward, 3, weightsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForward, 4, biasesGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForward, 5, (IntPtr)sizeof(int), nFilters);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForward, 6, (IntPtr)sizeof(int), receptiveFieldSize);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForward, 7, (IntPtr)sizeof(int), nReceptiveFields);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.FeedForward(): Cl.SetKernelArg ConvForward");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.ConvForward,
                                                                2,
                                                                null,
                                                                forwardGlobalWorkSizePtr,
                                                                forwardLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.FeedForward(): Cl.EnqueueNDRangeKernel ConvForward");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

#else
            // TODO: cpu code
#endif
            }

#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#endif
        }

        public override void BackPropagate()
        {

        }


        public override void UpdateSpeeds(double learningRate, double momentumCoefficient)
        {
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
#if OPENCL_ENABLED
                // Set kernel arguments
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeeds, 0, weightsSpeedGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeeds, 1, biasesSpeedGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeeds, 2, OutputNeurons.DeltaGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeeds, 3, InputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeeds, 4, receptiveFieldsLookupTableGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeeds, 5, (IntPtr)sizeof(int), nFilters);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeeds, 6, (IntPtr)sizeof(int), receptiveFieldSize);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeeds, 7, (IntPtr)sizeof(int), nReceptiveFields);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeeds, 8, (IntPtr)sizeof(float), (float)momentumCoefficient);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeeds, 9, (IntPtr)sizeof(float), (float)(learningRate / inputNeurons.MiniBatchSize));
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvUpdateSpeeds, 10, (IntPtr)sizeof(int), m);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.UpdateSpeeds(): Cl.SetKernelArg");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.FCUpdateSpeeds,
                                                                2,
                                                                null,
                                                                updateParametersGlobalWorkSizePtr,
                                                                updateParametersLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.UpdateSpeeds(): Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
#else
                // TODO: cpu code for speeds update
#endif
            }
        }

        public override void UpdateParameters()
        {
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
#else
            for (int iFilter = 0; iFilter < nFilters; iFilter++)
            {
                // weights update

                for (int iElement = 0; iElement < receptiveFieldSize; iElement++)
                {
                    weights[iFilter, iElement] += weightsUpdateSpeed[iFilter, iElement];
                }

                // update biases
                biases[iFilter] += biasesUpdateSpeed[iFilter];
            }
#endif
        }

        #endregion


        #region Obsolete methods

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
