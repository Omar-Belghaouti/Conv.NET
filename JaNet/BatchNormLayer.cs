using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCL.Net;

namespace JaNet
{
    class BatchNormLayer : Layer
    {

        #region Fields
        
        private string previousLayerType;
        private bool isEpochBeginning;
        private bool isTraining;
        private int iCumulativeAverage;


#if OPENCL_ENABLED
        // OpenCL-only fields

        private Mem meanGPU;
		private Mem varianceGPU;

		private Mem cumulativeMeanGPU;
		private Mem cumulativeVarianceGPU;

        private Mem meanGradientGPU;
        private Mem varianceGradientGPU;

        private Mem normalizedInputGPU;

        private Mem betaGPU;
        private Mem gammaGPU;

        private Mem betaSpeedGPU;
        private Mem gammaSpeedGPU;

        // Work group sizes
        private IntPtr[] nStatisticsGlobalWorkSizePtr;
        private IntPtr[] nStatisticsLocalWorkSizePtr;
        private IntPtr[] nParametersGlobalWorkSizePtr;

        private IntPtr[] nActivationsGlobalWorkSizePtr;
        private IntPtr[] nActivationsLocalWorkSizePtr;

#else
        // Host-only fields
        // TODO...

#endif
        #endregion


        #region Properties

        public override bool IsEpochBeginning
        {
            set { this.isEpochBeginning = value; }
        }
        public override bool IsTraining
        {
            set { this.isTraining = value; }
        }

        #endregion


        #region Setup methods

        /// <summary>
        /// Constructor of BatchNormLayer. Specify type of previous layer.
        /// </summary>
        /// <param name="PreviousLayerType"></param>
        public BatchNormLayer(string PreviousLayerType)
        {
            this.type = "BatchNorm";

            if (PreviousLayerType != "Convolutional" && PreviousLayerType != "FullyConnected")
                throw new ArgumentException("BatchNorm layer constructor: can only pass ''Convolutional'' or ''FullyConnected''.");
            else
                this.previousLayerType = PreviousLayerType;
        }

        public override void SetupOutput()
        {
            this.outputWidth = inputWidth;
            this.outputHeight = inputHeight;
            this.outputDepth = inputDepth;

            this.nOutputUnits = nInputUnits;
            this.outputNeurons = new Neurons(nOutputUnits);
        }

        public override void InitializeParameters()
        {
            this.iCumulativeAverage = 0;

#if OPENCL_ENABLED

            // Initialize OpenCL buffers for means and variances. Size depends on type of previous layer.
            int bufferSize = sizeof(float) * inputDepth; // default case: (previousLayerType == "Convolutional")
            if (previousLayerType == "FullyConnected")
                bufferSize = sizeof(float) * nInputUnits;

            this.meanGPU = (Mem)Cl.CreateBuffer(   OpenCLSpace.Context,
                                                    MemFlags.ReadWrite,
                                                    (IntPtr)bufferSize,
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

		    this.varianceGPU= (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                    MemFlags.ReadWrite,
                                                    (IntPtr)bufferSize,
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

		    this.cumulativeMeanGPU= (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)bufferSize,
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

		    this.cumulativeVarianceGPU= (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)bufferSize,
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.meanGradientGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)bufferSize,
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.varianceGradientGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)bufferSize,
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            // Write zeros, just in case
            OpenCLSpace.WipeBuffer(meanGPU, inputDepth, typeof(float));
            OpenCLSpace.WipeBuffer(varianceGPU, inputDepth, typeof(float));
            OpenCLSpace.WipeBuffer(cumulativeMeanGPU, inputDepth, typeof(float));
            OpenCLSpace.WipeBuffer(cumulativeVarianceGPU, inputDepth, typeof(float));
            OpenCLSpace.WipeBuffer(meanGradientGPU, inputDepth, typeof(float));
            OpenCLSpace.WipeBuffer(varianceGradientGPU, inputDepth, typeof(float));

            // Initialize OpenCL buffers for normalized input values (needed for backprop)
            this.normalizedInputGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * nInputUnits * inputNeurons.MiniBatchSize),
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(normalizedInputGPU, nInputUnits * inputNeurons.MiniBatchSize, typeof(float));


            // Initialize OpenCL buffers for learnable parameters and their update speed.
            // Write ones in gammas and zeros in betas (identity function in the beginning). Write zeros in speeds.

            int nParameters = inputDepth; // default case: (previousLayerType == "Convolutional")
            if (previousLayerType == "FullyConnected")
                nParameters = nInputUnits;

            float[] ones = new float[nParameters];
            for (int i = 0; i < nParameters; ++i)
                ones[i] = 1.0f;
            this.gammaGPU = (Mem)Cl.CreateBuffer  (OpenCLSpace.Context,
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                    (IntPtr)(sizeof(float) * nParameters),
                                                    ones,
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            float[] zeros = new float[nParameters];
            this.betaGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                (IntPtr)(sizeof(float) * nParameters),
                                                zeros,
                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.gammaSpeedGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * nParameters),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.betaSpeedGPU = (Mem)Cl.CreateBuffer(   OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * nParameters),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

#else
            // TODO... (cpu code)
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

            // 1D worksize equal to the size of means, variances, gamma, beta,... i.e. inputDepth if following a Conv layer, nInputUnits if following a FC layer
            // Use for ComputeMeansVariances(), UpdateSpeeds() and UpdateParameters() _____________________________________________________________________
            
            // Local
            this.nStatisticsLocalWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.BASE_GROUP_SIZE };

            // Global
            int workItemsNeeded = (previousLayerType == "Convolutional") ? inputDepth : nInputUnits;
            int smallestMultiple = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(workItemsNeeded) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.nStatisticsGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple };
            this.nParametersGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(2*smallestMultiple) }; // use for gradients kernel

            // 1D worksize equal to the total number of activations. Use for FeedForward() and BackPropagate() kernels ___________________

            // Local
            this.nActivationsLocalWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.OPTIMAL_GROUP_SIZE };
            
            // Global
            smallestMultiple = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE * Math.Ceiling((double)(nInputUnits * inputNeurons.MiniBatchSize) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
            this.nActivationsGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple};

            
#endif
        }

        #endregion

        #region Methods

        public override void FeedForward()
        {
            if (isEpochBeginning)
            {
                iCumulativeAverage = 0;
                isEpochBeginning = false;
            }

            if (previousLayerType == "Convolutional")
            {
                // Case 1 : this layer follows a Convolutional layer

#if OPENCL_ENABLED

                // 1.1) If training, compute means and variances, and update cumulative averages
                if (isTraining)
                {
                    OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 0, meanGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 1, varianceGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 2, cumulativeMeanGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 3, cumulativeVarianceGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 4, inputNeurons.ActivationsGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 5, (IntPtr)sizeof(int), inputDepth);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 6, (IntPtr)sizeof(int), inputHeight * inputWidth);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 7, (IntPtr)sizeof(int), nInputUnits);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 8, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 9, (IntPtr)sizeof(int), iCumulativeAverage);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                    OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                    OpenCLSpace.BNConvComputeMeansVariances,
                                                                    1,
                                                                    null,
                                                                    nStatisticsGlobalWorkSizePtr,
                                                                    nStatisticsLocalWorkSizePtr,
                                                                    0,
                                                                    null,
                                                                    out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                    OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                    // increase average counter
                    iCumulativeAverage++;
                }

                // 1.2) Normalize input, scale and shift

                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.BNConvForward, 0, outputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 1, normalizedInputGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 2, inputNeurons.ActivationsGPU);
                if (isTraining)
                {
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 3, meanGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 4, varianceGPU);
                }
                else
                {
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 3, cumulativeMeanGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 4, cumulativeVarianceGPU);
                }
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 5, gammaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 6, betaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 7, (IntPtr)sizeof(int), inputHeight * inputWidth);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 8, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 9, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                                OpenCLSpace.BNConvForward,
                                                                1,
                                                                null,
                                                                nActivationsGlobalWorkSizePtr,
                                                                nActivationsLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            // TODO (cpu)
#endif
            }
            else // case 2 : previousLayerType == "FullyConnected"
            {
#if OPENCL_ENABLED

                // 2.1) If training, compute means and variances, and update cumulative averages
                if (isTraining)
                {
                    OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 0, meanGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 1, varianceGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 2, cumulativeMeanGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 3, cumulativeVarianceGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 4, inputNeurons.ActivationsGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 5, (IntPtr)sizeof(int), nInputUnits);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 6, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 7, (IntPtr)sizeof(int), iCumulativeAverage);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                    OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                    OpenCLSpace.BNFCComputeMeansVariances,
                                                                    1,
                                                                    null,
                                                                    nStatisticsGlobalWorkSizePtr,
                                                                    nStatisticsLocalWorkSizePtr,
                                                                    0,
                                                                    null,
                                                                    out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                    OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                    // increase average counter
                    iCumulativeAverage++;
                }

#if DEBUGGING_STEPBYSTEP

                /* ------------------------- DEBUGGING --------------------------------------------- */

                // Display means
                float[] means = new float[nInputUnits];

                OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                            meanGPU, // source
                                                            Bool.True,
                                                            (IntPtr)0,
                                                            (IntPtr)(nInputUnits * sizeof(float)),
                                                            means,  // destination
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");
                    
                Console.WriteLine("\nMeans:\n");
                for (int i = 0; i < nInputUnits; i++)
                    Console.Write("{0}  ", means[i]);
                Console.ReadKey();

                // Display variances
                float[] var = new float[nInputUnits];

                OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                            varianceGPU, // source
                                                            Bool.True,
                                                            (IntPtr)0,
                                                            (IntPtr)(nInputUnits * sizeof(float)),
                                                            var,  // destination
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

                Console.WriteLine("\n\nVariances:\n");
                for (int i = 0; i < nInputUnits; i++)
                    Console.Write("{0}  ", var[i]);
                Console.ReadKey();

                 
                /* ------------------------- END DEBUGGING --------------------------------------------- */

#endif


                // 1.2) Normalize input, scale and shift

                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNFCForward, 0, outputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 1, normalizedInputGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 2, inputNeurons.ActivationsGPU);
                if (isTraining)
                {
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 3, meanGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 4, varianceGPU);
                }
                else
                {
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 3, cumulativeMeanGPU);
                    OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 4, cumulativeVarianceGPU);
                }
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 5, gammaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 6, betaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 7, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 8, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.BNFCForward,
                                                                1,
                                                                null,
                                                                nActivationsGlobalWorkSizePtr,
                                                                nActivationsLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            // TODO (cpu)
#endif
            }
            
            
            
        }


        public override void BackPropagate()
        {
            
            if (previousLayerType == "Convolutional")
            {
                // Case 1 : this layer follows a Convolutional layer
#if OPENCL_ENABLED

                // 1.1) Compute gradients of loss function wrt mean and variance in each feature map

                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNConvGradientMeanVariance, 0, meanGradientGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvGradientMeanVariance, 1, varianceGradientGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvGradientMeanVariance, 2, outputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvGradientMeanVariance, 3, normalizedInputGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvGradientMeanVariance, 4, gammaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvGradientMeanVariance, 5, varianceGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvGradientMeanVariance, 6, (IntPtr)sizeof(int), inputDepth);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvGradientMeanVariance, 7, (IntPtr)sizeof(int), inputHeight*inputWidth);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvGradientMeanVariance, 8, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.BNConvGradientMeanVariance,
                                                                1,
                                                                null,
                                                                nParametersGlobalWorkSizePtr,
                                                                nStatisticsLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                // 1.2) Backpropagate to input deltas using above gradients as auxiliary variables

                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 0, inputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 1, outputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 2, inputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 3, gammaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 4, meanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 5, varianceGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 6, meanGradientGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 7, varianceGradientGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 8, (IntPtr)sizeof(int), inputHeight * inputWidth);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 9, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 10, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.BNConvBackPropagate,
                                                                1,
                                                                null,
                                                                nActivationsGlobalWorkSizePtr,
                                                                nActivationsLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            // TODO: cpu code
#endif
            }
            else 
            {
                // Case 2 : previousLayerType == "FullyConnected"

#if OPENCL_ENABLED

                // 2.1) Compute gradients of loss function wrt mean and variance in each feature map

                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 0, meanGradientGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 1, varianceGradientGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 2, outputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 3, normalizedInputGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 4, gammaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 5, varianceGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 6, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 7, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.BNFCGradientMeanVariance,
                                                                1,
                                                                null,
                                                                nParametersGlobalWorkSizePtr,
                                                                nStatisticsLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                // 2.2) Backpropagate to input deltas using above gradients as auxiliary variables

                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 0, inputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 1, outputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 2, inputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 3, gammaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 4, meanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 5, varianceGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 6, meanGradientGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 7, varianceGradientGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 8, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 9, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.BNFCBackPropagate,
                                                                1,
                                                                null,
                                                                nActivationsGlobalWorkSizePtr,
                                                                nActivationsLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            // TODO: cpu code
#endif
            }
        }

        public override void UpdateSpeeds(double learningRate, double momentumMultiplier)
        {
            
            if (previousLayerType == "Convolutional")
            {
                // Case 1 : this layer follows a Convolutional layer
#if OPENCL_ENABLED
                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 0, gammaSpeedGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 1, betaSpeedGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 2, outputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 3, normalizedInputGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 4, (IntPtr)sizeof(int), inputDepth);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 5, (IntPtr)sizeof(int), inputHeight * inputWidth);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 6, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 7, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 8, (IntPtr)sizeof(float), (float)momentumMultiplier);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 9, (IntPtr)sizeof(float), (float)learningRate);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                                OpenCLSpace.BNConvUpdateSpeeds,
                                                                1,
                                                                null,
                                                                nParametersGlobalWorkSizePtr,
                                                                nStatisticsLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

#else
            // TODO: cpu code
#endif
            }
            else 
            {
                // Case 2 : previousLayerType == "FullyConnected"
#if OPENCL_ENABLED
                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 0, gammaSpeedGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 1, betaSpeedGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 2, outputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 3, normalizedInputGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 4, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 5, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 6, (IntPtr)sizeof(float), (float)momentumMultiplier);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 7, (IntPtr)sizeof(float), (float)learningRate);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.BNFCUpdateSpeeds,
                                                                1,
                                                                null,
                                                                nParametersGlobalWorkSizePtr,
                                                                nStatisticsLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

#else
            // TODO: cpu code
#endif

            }
        }


        public override void UpdateParameters(double weightDecayCoeff)
        {
            // In this case the only difference between convolutional layer and FC layer is in how the last kernel argument 
            // is set (and in the work group sizes, set previously)

#if OPENCL_ENABLED
            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNUpdateParameters, 0, gammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNUpdateParameters, 1, betaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNUpdateParameters, 2, gammaSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNUpdateParameters, 3, betaSpeedGPU);
            if (previousLayerType == "Convolutional")
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNUpdateParameters, 4, (IntPtr)sizeof(int), inputDepth);
            else
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNUpdateParameters, 4, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.BNUpdateParameters,
                                                            1,
                                                            null,
                                                            nStatisticsGlobalWorkSizePtr,
                                                            nStatisticsLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");


#if DEBUGGING_STEPBYSTEP

            /* ------------------------- DEBUGGING --------------------------------------------- */

            // Display gamma
            float[] gamma = new float[nInputUnits];

            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        gammaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(nInputUnits * sizeof(float)),
                                                        gamma,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

            Console.WriteLine("\nUpdated gammas are:\n");
            for (int i = 0; i < nInputUnits; i++)
                Console.Write("{0}  ", gamma[i]);
            Console.ReadKey();

            // Display beta
            float[] beta = new float[nInputUnits];

            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        betaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(nInputUnits * sizeof(float)),
                                                        beta,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

            Console.WriteLine("\n\nUpdated betas are:\n");
            for (int i = 0; i < nInputUnits; i++)
                Console.Write("{0}  ", beta[i]);
            Console.ReadKey();


            /* ------------------------- END DEBUGGING --------------------------------------------- */

#endif

#else
            // TODO: cpu code
#endif

        }

        #endregion

    }
}
