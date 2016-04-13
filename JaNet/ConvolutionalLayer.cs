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

        //private IntPtr[] weightsUpdateSpeedGlobalWorkSizePtr;
        //private IntPtr[] weightsUpdateSpeedLocalWorkSizePtr;

        //private IntPtr[] biasesUpdateSpeedGlobalWorkSizePtr;
        //private IntPtr[] biasesUpdateSpeedLocalWorkSizePtr;

        private IntPtr[] updateParametersGlobalWorkSizePtr;
        private IntPtr[] updateParametersLocalWorkSizePtr;

#else
        private double[] paddedInput; // dimension [inputD * (inputH + 2*padding) * (inutW + 2*padding)]
        private int[,] lookupTable; // dimension [receptiveFieldSize , nReceptiveFields] = [inputDepth*filterSize^2 , outputWidth*outputHeight]
        

        private double[,] weights; // dimension [nFilters , inputDepth*filterSize^2]
        private double[] biases; // dimension [nFilters , 1]

#if GRADIENT_CHECK
        private double[,] weightsGradients;
        private double[] biasesGradients;
#endif

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


        public override void ConnectTo(Layer PreviousLayer)
        {
            // Setup input
            base.ConnectTo(PreviousLayer);

            if (PreviousLayer.OutputHeight != PreviousLayer.OutputWidth)
                throw new ArgumentException("ConvolutionalLayer currently only supports square input (spatially).");

            

            // Setup output
            double tmp = (double)(inputWidth - filterSize + 2 * zeroPadding) / (double)strideLength + 1; // then check if this number is int
            if (Math.Abs(tmp % 1) > Global.EPSILON) 
                throw new System.ArgumentException("Input width, filter size, zero padding and stride length do not fit well. Use different values");
            this.outputWidth = (int)tmp;
            this.outputHeight = (int)tmp;
            this.nReceptiveFields = outputHeight * outputWidth;

            this.outputDepth = nFilters;
            this.receptiveFieldSize = inputDepth * filterSize * filterSize;

            this.nOutputUnits = outputDepth * outputWidth * outputHeight;
            this.outputNeurons = new Neurons(outputDepth * outputWidth * outputHeight);

            this.inputArea = inputWidth * inputHeight;
            this.inputVolume = inputWidth * inputHeight * inputDepth;
            this.nZerosTopRows = zeroPadding * (2 * zeroPadding + inputWidth);
            this.nZerosPerSlice = 2 * zeroPadding * (inputWidth + inputHeight + 2 * zeroPadding);

            this.paddedInputSize = inputDepth * (inputHeight + 2 * zeroPadding) * (inputWidth + 2 * zeroPadding);
            this.receptiveFieldsLookupTableSize = receptiveFieldSize * nReceptiveFields;

#if OPENCL_ENABLED

            if (zeroPadding > 0)
            {
                // Padded input buffer

                this.paddedInputGPU = new List<Mem>();
                this.paddedInputGPU.Add( (Mem)Cl.CreateBuffer(  OpenCLSpace.Context, 
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)(sizeof(float) * paddedInputSize),
                                                                out OpenCLSpace.ClError) );
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ConnectTo(): Cl.CreateBuffer paddedInputGPU");
            }

            // Receptive fields lookup table buffer

            this.receptiveFieldsLookupTableGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                                        MemFlags.ReadWrite,
                                                                        (IntPtr)(sizeof(int) * receptiveFieldsLookupTableSize),
                                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ConnectTo(): Cl.CreateBuffer receptiveFieldsLookupTableGPU");

            

            // (no need for output matrix: will be written directly to OuptutNeurons.ActivationsGPU
#else
            // Cpu code

            this.paddedInput = new double[paddedInputSize];
            this.lookupTable = new int[receptiveFieldSize, nReceptiveFields];
#endif

           
#if OPENCL_ENABLED
            // Set all work group sizes
            SetWorkGroupSizes();

            // We're ready to create the lookup table once and for all

            // Set kernel arguments
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.Im2colLookupTable, 0, receptiveFieldsLookupTableGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.Im2colLookupTable, 1, (IntPtr)sizeof(int), inputWidth + 2*zeroPadding);
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
#else
            lookupTable = CreateLookupTableCPU();
#endif
        }

#if OPENCL_ENABLED
        private void SetWorkGroupSizes()
        {
            // Work group sizes will be set as follows:
            //      global work size = smallest multiple of BASE_GROUP_SIZE larger than the total number of processes needed (for efficiency)
            //      local work size = BASE_GROUP_SIZE or small multiples of it (making sure that global worksize is a multiple of this)
            // BASE_GROUP_SIZE is a constant multiple of 2. Suggested values: 32 (Nvidia) or 64 (AMD).

            // Zero padding / unpadding (1D workspace) _________________________________________________________________________________
            if (zeroPadding > 0)
            {
                int inputSize = inputDepth * inputWidth * inputHeight;
                int smallestMultipleInputSize = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(inputSize) / (double)OpenCLSpace.BASE_GROUP_SIZE));
                this.paddingGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultipleInputSize };
                int localWorkSize = OpenCLSpace.BASE_GROUP_SIZE;
                int maxKernelWorkGroupSize = Cl.GetKernelWorkGroupInfo(OpenCLSpace.ZeroPad,
                                                                        OpenCLSpace.Device,
                                                                        KernelWorkGroupInfo.WorkGroupSize,
                                                                        out OpenCLSpace.ClError).CastTo<int>();
                while (true)
                {
                    int tmpLocalWorkSize = 2 * localWorkSize;
                    bool globalDividesLocal = smallestMultipleInputSize % tmpLocalWorkSize == 0;
                    bool isLocalGroupTooLarge = tmpLocalWorkSize > OpenCLSpace.MaxWorkGroupSize;
                    isLocalGroupTooLarge |= tmpLocalWorkSize > maxKernelWorkGroupSize;

                    if (globalDividesLocal && !isLocalGroupTooLarge) // if global divides local and it's not too large
                        localWorkSize = tmpLocalWorkSize;
                    else
                        break;
                }
                this.paddingLocalWorkSizePtr = new IntPtr[] { (IntPtr)localWorkSize };
            }

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

            for (int iRow = 0; iRow < initWeights.GetLength(0); iRow++)
            {
                for (int iCol = 0; iCol < initWeights.GetLength(1); iCol++)
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

            this.weightsSpeedGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)weightBufferSize,
                                                                out OpenCLSpace.ClError);
            this.biasesSpeedGPU = (Mem)Cl.CreateBuffer(   OpenCLSpace.Context,
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
#endif

#if GRADIENT_CHECK
            this.weightsGradients = new double[nFilters, receptiveFieldSize];
            this.biasesGradients = new double[nFilters];
#endif
        }

        #endregion


        #region Training methods
        

        public override void FeedForward()
        {
#if OPENCL_ENABLED
            if (zeroPadding > 0)
            {
                // inelegant workaround to add memory buffers to List<Mem> paddedInputGPU in case miniBatchSize > 1
                // TODO: find a more elegant solution for this
                while (inputNeurons.MiniBatchSize > paddedInputGPU.Count)
                {
                    this.paddedInputGPU.Add((Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                    MemFlags.ReadWrite,
                                                                    (IntPtr)(sizeof(float) * paddedInputSize),
                                                                    out OpenCLSpace.ClError));
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ConnectTo(): Cl.CreateBuffer paddedInputGPU");
                }
            }

            // Forward method begins here

            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                if (zeroPadding > 0)
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
                    OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
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
                }

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

                // 2. Convolve input and filters _________________________________________________________

                // Set kernel arguments
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ConvForward, 0, outputNeurons.ActivationsGPU[m]);
                if (zeroPadding > 0)
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForward, 1, paddedInputGPU[m]);
                else
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvForward, 1, inputNeurons.ActivationsGPU[m]);
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
            }
#else
            // TODO: cpu code
            if (inputNeurons.MiniBatchSize > 1)
                throw new ArgumentException("Only online-training is supported if using CPU.");

            // Padding
            if (zeroPadding > 0)
                paddedInput = ZeroPadCPU(inputNeurons.GetHost()[0], zeroPadding, inputDepth, inputHeight, inputWidth);

            // Forward (matrix multiplication)
            outputNeurons.SetHost(0, ConvForwardCPU(paddedInput));



#endif

#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#endif
        }

        public override void BackPropagate()
        {
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
#if OPENCL_ENABLED
                if (zeroPadding > 0)
                {
                    // 1. Wipe out paddedInput buffer (will be cumulated!)
                    OpenCLSpace.WipeBuffer(paddedInputGPU[m], paddedInputSize, typeof(float));

                    OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
                }

                // 2. BackPropagate kernel
                
                // Set kernel arguments
                if (zeroPadding > 0)
                    OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ConvBackPropagate, 0, paddedInputGPU[m]);
                else
                    OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ConvBackPropagate, 0, inputNeurons.DeltaGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagate, 1, outputNeurons.DeltaGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagate, 2, weightsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagate, 3, receptiveFieldsLookupTableGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagate, 4, (IntPtr)sizeof(int), nFilters);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagate, 5, (IntPtr)sizeof(int), receptiveFieldSize);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ConvBackPropagate, 6, (IntPtr)sizeof(int), nReceptiveFields);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.BackPropagate(): Cl.SetKernelArg ConvBackPropagate");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.ConvBackPropagate,
                                                                2,
                                                                null,
                                                                backwardGlobalWorkSizePtr,
                                                                backwardLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.BackPropagate(): Cl.EnqueueNDRangeKernel ConvBackPropagate");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                if (zeroPadding > 0)
                {
                    OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

                    // 3. Unpad paddedInputGPU[m] to get deltaX

                    // Set kernel arguments
                    OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ZeroUnpad, 0, paddedInputGPU[m]);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpad, 1, inputNeurons.DeltaGPU[m]);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpad, 2, (IntPtr)sizeof(int), inputWidth);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpad, 3, (IntPtr)sizeof(int), inputArea);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpad, 4, (IntPtr)sizeof(int), inputVolume);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpad, 5, (IntPtr)sizeof(int), zeroPadding);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpad, 6, (IntPtr)sizeof(int), nZerosTopRows);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ZeroUnpad, 7, (IntPtr)sizeof(int), nZerosPerSlice);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.BackPropagate(): Cl.SetKernelArg ZeroUnpad");

                    // Run kernel
                    OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                                    OpenCLSpace.ZeroUnpad,
                                                                    1,
                                                                    null,
                                                                    paddingGlobalWorkSizePtr,
                                                                    paddingLocalWorkSizePtr,
                                                                    0,
                                                                    null,
                                                                    out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Convolutional.BackPropagate(): Cl.EnqueueNDRangeKernel ZeroUnpad");

                    OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
                }
#else
            //TODO: cpu code for backprop.
#endif
            }

#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#endif



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
                if (inputNeurons.MiniBatchSize > 1)
                    throw new ArgumentException("Only miniBatchSize = 1 is currently supported if using CPU");

                ConvGradientsCPU(ref weightsGradients, ref biasesGradients, outputNeurons.DeltaHost[0], paddedInput);

#endif
            }
#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#endif
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


        private static double[] ZeroUnpadCPU(double[] paddedArray, int padding, int inputDepth, int inputHeight, int inputWidth)
        {
            int area = inputHeight * inputWidth;
            int volume = inputDepth * inputHeight * inputWidth;
            int nZerosTopRows = padding * (2 * padding + inputWidth);
            int zerosPerSlice = 2 * padding * (inputHeight + inputWidth + 2 * padding);

            double[] unpaddedArray = new double[volume];

            // auxiliary variables
            int iRow, iSlice, iPadded;

            for (int iUnpadded = 0; iUnpadded < volume; iUnpadded++)
            {
                iSlice = (iUnpadded % volume) / area; // find index of channel within an input volume
                iRow = (iUnpadded % area) / inputWidth; // find index of row within an input channel
                iPadded = zerosPerSlice * iSlice + nZerosTopRows + padding * (2 * iRow + 1) + iUnpadded;

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
