using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCL.Net;

namespace JaNet
{
#if OPENCL_ENABLED

    [Serializable]
    class BatchNormConv : Layer
    {

        #region Fields

        private bool isEpochBeginning;
        private bool isTraining;
        private bool isPreInference;
        private bool isInference;

        private int iCumulativeAverage;
        private int inputArea;

        // Host parameters

        private float[] gammaHost;
        private float[] betaHost;

        // Device

        [NonSerialized]
        private Mem meanGPU;
        [NonSerialized]
        private Mem varianceGPU;

        [NonSerialized]
        private Mem cumulativeMeanGPU;
        [NonSerialized]
        private Mem cumulativeVarianceGPU;

        [NonSerialized]
        private Mem normalizedInputGPU;

        [NonSerialized]
        private Mem gammaGPU;
        [NonSerialized]
        private Mem betaGPU;

        [NonSerialized]
        private Mem deltaGammaBatchGPU;
        [NonSerialized]
        private Mem deltaBetaBatchGPU;

        [NonSerialized]
        private Mem deltaGammaGPU;
        [NonSerialized]
        private Mem deltaBetaGPU;

        [NonSerialized]
        private Mem gammaSpeedGPU;
        [NonSerialized]
        private Mem betaSpeedGPU;

        // Work group sizes
        private IntPtr[] optimalLocalWorkSizePtr;
        private IntPtr[] baseLocalWorkSizePtr;

        private IntPtr[] nFeatureMapsGlobalWorkSizePtr;
        private IntPtr[] nActivationsGlobalWorkSizePtr; // = nUnits * miniBatchSize
        private IntPtr[] nUnitsGlobalWorkSizePtr;


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
        public override bool IsPreInference
        {
            set { this.isPreInference = value; }
        }
        public override bool IsInference
        {
            set { this.isInference = value; }
        }


        #endregion


        #region Setup methods

        /// <summary>
        /// Constructor of BatchNormConv Layer.
        /// </summary>
        public BatchNormConv()
        {
            this.type = "BatchNormConv";
        }

        public override void SetupOutput()
        {
            this.outputWidth = inputWidth;
            this.outputHeight = inputHeight;
            this.outputDepth = inputDepth;
            this.inputArea = inputHeight * inputWidth;

            this.nOutputUnits = nInputUnits;
            this.outputNeurons = new Neurons(nOutputUnits);

            // Initialize OpenCL buffers

            // 1. mean, variance and their cumulative averages

            this.meanGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                MemFlags.ReadWrite,
                                                (IntPtr)(sizeof(float) * inputDepth),
                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(meanGPU, inputDepth, typeof(float));

            this.varianceGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                    MemFlags.ReadWrite,
                                                    (IntPtr)(sizeof(float) * inputDepth),
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(varianceGPU, inputDepth, typeof(float));

            // (Initialize cumulative means to zero...)
            this.cumulativeMeanGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * inputDepth),
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(cumulativeMeanGPU, inputDepth, typeof(float));

            // (...and variances to one.)
            float[] ones = new float[inputDepth];
            for (int i = 0; i < inputDepth; ++i)
                ones[i] = 1.0f;
            this.cumulativeVarianceGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                                (IntPtr)(sizeof(float) * inputDepth),
                                                                ones,
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            // OpenCL buffer for normalized input values (needed for backprop)

            this.normalizedInputGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * nInputUnits * inputNeurons.MiniBatchSize),
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(normalizedInputGPU, nInputUnits * inputNeurons.MiniBatchSize, typeof(float));
        }

        public override void InitializeParameters(string Option)
        {
            this.iCumulativeAverage = 0;
            this.isEpochBeginning = true;

            this.isTraining = true;
            this.isPreInference = false;
            this.isInference = false;

            // Initialize OpenCL buffers for learnable parameters gamma and beta, their gradients, and their update speed.

            if (Option == "random")
            {
                // A new network is being created. Write ones in gammas and zeros in betas (identity function in the beginning).
                this.gammaHost = new float[inputDepth];
                for (int i = 0; i < inputDepth; ++i)
                    gammaHost[i] = 1.0f;

                this.betaHost = new float[inputDepth];

            }
            // else Option must be "load", in which case just copy host values (retrieved from file) to device

            this.gammaGPU = (Mem)Cl.CreateBuffer(   OpenCLSpace.Context,
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                    (IntPtr)(sizeof(float) * inputDepth),
                                                    gammaHost,
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.betaGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                (IntPtr)(sizeof(float) * inputDepth),
                                                betaHost,
                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.deltaGammaBatchGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * nInputUnits),
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(deltaGammaBatchGPU, nInputUnits, typeof(float));

            this.deltaBetaBatchGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * nInputUnits),
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(deltaBetaBatchGPU, nInputUnits, typeof(float));

            this.deltaGammaGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * inputDepth),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(deltaGammaGPU, inputDepth, typeof(float));

            this.deltaBetaGPU = (Mem)Cl.CreateBuffer(   OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * inputDepth),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(deltaBetaGPU, inputDepth, typeof(float));

            this.gammaSpeedGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * inputDepth),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(gammaSpeedGPU, inputDepth, typeof(float));

            this.betaSpeedGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * inputDepth),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(betaSpeedGPU, inputDepth, typeof(float));
        }


        public override void SetWorkGroups()
        {

            // Work group sizes will be set as follows:
            //      global work size = smallest multiple of OPTIMAL_GROUP_SIZE larger than 
            //                         the total number of processes needed (for efficiency).
            //      local work size = as close as possible to OPTIMAL_GROUP_SIZE (making sure 
            //                        that global worksize is a multiple of this)
            // OPTIMAL_GROUP_SIZE is a small multiple of BASE_GROUP_SIZE, which in turn is a 
            //                    constant multiple of 2, platform-dependent, e.g. 32 (Nvidia 
            //                    WARP) or 64 (AMD WAVEFRONT).

            // Local
            this.baseLocalWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.BASE_GROUP_SIZE };
            this.optimalLocalWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.OPTIMAL_GROUP_SIZE };

            // _____________________________________________________________________________________________________________________
            // Global 1D worksize of size inputDepth: Use for ComputeMeansVariances(), UpdateParameters(), and UpdateSpeeds() 

            int smallestMultiple = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(inputDepth) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.nFeatureMapsGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple };

            // _____________________________________________________________________________________________________________________
            // Global 1D worksize of size 2*inputDepth: Use for UpdateSpeeds() 
            //this.nParametersGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(2 * smallestMultiple) };

            // _____________________________________________________________________________________________________________________
            // Global 1D worksize equal to the total number of activations (i.e. input tensor size). Use for FeedForward() and BackPropagate() kernels 

            // Global
            smallestMultiple = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE * Math.Ceiling((double)(nInputUnits * inputNeurons.MiniBatchSize) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
            this.nActivationsGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple };

            // _____________________________________________________________________________________________________________________
            // Global 1D worksize equal to the number of units (not accounting for mini-batch size). Use for BNConvParameterGradientsBatch() kernel

            // Global
            smallestMultiple = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE * Math.Ceiling((double)(nInputUnits) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
            this.nUnitsGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple };

            
        }


        public override void CopyBuffersToHost()
        {
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        gammaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputDepth),
                                                        gammaHost,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer gammaGPU");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        betaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputDepth),
                                                        betaHost,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer betaGPU");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            // Speeds are not saved.
        }

        #endregion


        #region Methods

        public override void FeedForward()
        {
#if TIMING_LAYERS
            Utils.BNConvForwardTimer.Start();
#endif
            if (isEpochBeginning)
            {
                iCumulativeAverage = 0; 

                // Wipe cumulative means and variances
                OpenCLSpace.WipeBuffer(cumulativeMeanGPU, inputDepth, typeof(float));
                OpenCLSpace.WipeBuffer(cumulativeVarianceGPU, inputDepth, typeof(float));

                isEpochBeginning = false;
            }

            // If training, compute means and variances. If pre-inference, compute means and variances and also update cumulative averages
            if (isTraining || isPreInference)
            {
                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 0, meanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 1, varianceGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 2, cumulativeMeanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 3, cumulativeVarianceGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 4, inputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 5, (IntPtr)sizeof(int), inputDepth);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 6, (IntPtr)sizeof(int), inputArea);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 7, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 8, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 9, (IntPtr)sizeof(int), Convert.ToInt32(isPreInference));
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvComputeMeansVariances, 10, (IntPtr)sizeof(int), iCumulativeAverage);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");
            
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.BNConvComputeMeansVariances,
                                                                1,
                                                                null,
                                                                nFeatureMapsGlobalWorkSizePtr,
                                                                baseLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                if (isPreInference)
                {


                    /* ------------------------- DEBUGGING ---------------------------------------------

                    Console.WriteLine("\nPRE-INFERENCE MINI-BATCH {0}\n", iCumulativeAverage);
                    // Display cum means

                    float[] cumMeans = new float[inputDepth];

                    OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                                cumulativeMeanGPU, // source
                                                                Bool.True,
                                                                (IntPtr)0,
                                                                (IntPtr)(inputDepth * sizeof(float)),
                                                                cumMeans,  // destination
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

                    Console.WriteLine("\nCumulative means:\n");
                    for (int i = 0; i < inputDepth; i++)
                        Console.Write("{0}  ", cumMeans[i]);
                    //Console.ReadKey();

                    // Display cum var
                    float[] cumVar = new float[inputDepth];

                    OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                                cumulativeVarianceGPU, // source
                                                                Bool.True,
                                                                (IntPtr)0,
                                                                (IntPtr)(inputDepth * sizeof(float)),
                                                                cumVar,  // destination
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

                    Console.WriteLine("\n\nCumulative variance:\n");
                    for (int i = 0; i < inputDepth; i++)
                        Console.Write("{0}  ", cumVar[i]);
                    Console.ReadKey();


                    /* ------------------------- END DEBUGGING --------------------------------------------- */

                    iCumulativeAverage++; // increase cumulative average counter
                }
            }

            //Normalize input, scale and shift

            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.BNConvForward, 0, outputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 1, normalizedInputGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 2, inputNeurons.ActivationsGPU);
            if (isTraining || isPreInference)
            {
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 3, meanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 4, varianceGPU);
            }
            else if (isInference)
            {
                /* ------------------------- DEBUGGING ---------------------------------------------


                // Display cum means

                float[] cumMeans = new float[inputDepth];

                OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                            cumulativeMeanGPU, // source
                                                            Bool.True,
                                                            (IntPtr)0,
                                                            (IntPtr)(inputDepth * sizeof(float)),
                                                            cumMeans,  // destination
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

                Console.WriteLine("\nCumulative means:\n");
                for (int i = 0; i < inputDepth; i++)
                    Console.Write("{0}  ", cumMeans[i]);
                //Console.ReadKey();

                // Display cum var
                float[] cumVar = new float[inputDepth];

                OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                            cumulativeVarianceGPU, // source
                                                            Bool.True,
                                                            (IntPtr)0,
                                                            (IntPtr)(inputDepth * sizeof(float)),
                                                            cumVar,  // destination
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

                Console.WriteLine("\n\nCumulative variance:\n");
                for (int i = 0; i < inputDepth; i++)
                    Console.Write("{0}  ", cumVar[i]);
                Console.ReadKey();


                /* ------------------------- END DEBUGGING --------------------------------------------- */

                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 3, cumulativeMeanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 4, cumulativeVarianceGPU);
            }
            else
                throw new InvalidOperationException("ERROR: BatchNormConv is currently not in training mode, nor pre-inference, nor inference.");
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 5, gammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 6, betaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 7, (IntPtr)sizeof(int), inputArea);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 8, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvForward, 9, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");
            
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.BNConvForward,
                                                            1,
                                                            null,
                                                            nActivationsGlobalWorkSizePtr,
                                                            optimalLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

#if TIMING_LAYERS
            Utils.BNConvForwardTimer.Stop();
#endif

        }


        public override void BackPropagate()
        {

#if TIMING_LAYERS
            Utils.BNConvBackpropTimer.Start();
#endif

            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 0, inputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 1, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 2, normalizedInputGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 3, gammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 4, varianceGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 5, deltaGammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 6, deltaBetaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 7, (IntPtr)sizeof(int), inputArea);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 8, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvBackPropagate, 9, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.BNConvBackPropagate,
                                                            1,
                                                            null,
                                                            nActivationsGlobalWorkSizePtr,
                                                            optimalLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

#if TIMING_LAYERS
            Utils.BNConvBackpropTimer.Stop();
#endif
        }

        public override void UpdateSpeeds(double learningRate, double momentumMultiplier)
        {

#if TIMING_LAYERS
            Utils.BNConvUpdateSpeedsTimer.Stop();
#endif
            // Step 1
            
            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNConvParameterGradientsBatch, 0, deltaGammaBatchGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvParameterGradientsBatch, 1, deltaBetaBatchGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvParameterGradientsBatch, 2, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvParameterGradientsBatch, 3, normalizedInputGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvParameterGradientsBatch, 4, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvParameterGradientsBatch, 5, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                            OpenCLSpace.BNConvParameterGradientsBatch,
                                                            1,
                                                            null,
                                                            nUnitsGlobalWorkSizePtr,
                                                            optimalLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            
            // Step 2

            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 0, gammaSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 1, betaSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 2, deltaGammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 3, deltaBetaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 4, deltaGammaBatchGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 5, deltaBetaBatchGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 6, (IntPtr)sizeof(int), inputDepth);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 7, (IntPtr)sizeof(int), inputArea);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 8, (IntPtr)sizeof(float), (float)momentumMultiplier);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 9, (IntPtr)sizeof(float), (float)learningRate);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateSpeeds, 10, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.BNConvUpdateSpeeds,
                                                            1,
                                                            null,
                                                            nFeatureMapsGlobalWorkSizePtr,
                                                            baseLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
            

            /* ------------------------- DEBUGGING ---------------------------------------------
            
            
           // Display gamma gradient
            
           float[] deltaGgamma = new float[inputDepth];

           OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                       deltaGammaGPU, // source
                                                       Bool.True,
                                                       (IntPtr)0,
                                                       (IntPtr)(inputDepth * sizeof(float)),
                                                       deltaGgamma,  // destination
                                                       0,
                                                       null,
                                                       out OpenCLSpace.ClEvent);
           OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

           Console.WriteLine("\nGradient wrt gamma:\n");
           for (int i = 0; i < inputDepth; i++)
               Console.Write("{0}  ", deltaGgamma[i]);
           //Console.ReadKey();
            
           // Display beta
           float[] deltaBeta = new float[inputDepth];

           OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                       deltaBetaGPU, // source
                                                       Bool.True,
                                                       (IntPtr)0,
                                                       (IntPtr)(inputDepth * sizeof(float)),
                                                       deltaBeta,  // destination
                                                       0,
                                                       null,
                                                       out OpenCLSpace.ClEvent);
           OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

           Console.WriteLine("\n\nGradient wrt beta:\n");
           for (int i = 0; i < inputDepth; i++)
               Console.Write("{0}  ", deltaBeta[i]);
           Console.ReadKey();


           /* ------------------------- END DEBUGGING --------------------------------------------- */

            

#if TIMING_LAYERS
            Utils.BNConvUpdateSpeedsTimer.Stop();
#endif
             
        }


        public override void UpdateParameters(double weightDecayCoeff)
        {
            
#if TIMING_LAYERS
            Utils.BNConvUpdateParametersTimer.Start();
#endif
            
            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNConvUpdateParameters, 0, gammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateParameters, 1, betaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateParameters, 2, gammaSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateParameters, 3, betaSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNConvUpdateParameters, 4, (IntPtr)sizeof(int), inputDepth);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.BNConvUpdateParameters,
                                                            1,
                                                            null,
                                                            nFeatureMapsGlobalWorkSizePtr,
                                                            baseLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
            

            /* ------------------------- DEBUGGING ---------------------------------------------    
            
            
            // Display gamma
            
            float[] gamma = new float[inputDepth];

            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        gammaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(inputDepth * sizeof(float)),
                                                        gamma,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

            Console.WriteLine("\nUpdated gammas are:\n");
            for (int i = 0; i < inputDepth; i++)
                Console.Write("{0}  ", gamma[i]);
            //Console.ReadKey();
            
            // Display beta
            float[] beta = new float[inputDepth];

            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        betaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(inputDepth * sizeof(float)),
                                                        beta,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

            Console.WriteLine("\n\nUpdated betas are:\n");
            for (int i = 0; i < inputDepth; i++)
                Console.Write("{0}  ", beta[i]);
            Console.ReadKey();


            /* ------------------------- END DEBUGGING --------------------------------------------- */

#if TIMING_LAYERS
            Utils.BNConvUpdateParametersTimer.Stop();
#endif

        }

        #endregion

        #region Gradient check

        public override double[] GetParameters()
        {
            double[] parameters = new double[2 * inputDepth]; 

            // Copy gamma and beta buffers to host
            CopyBuffersToHost();

            // Convert to double and write into public fields
            for (int i = 0; i < inputDepth; ++i)
            {
                parameters[i] = (double)gammaHost[i];
                parameters[inputDepth + i] = (double)betaHost[i];
            }

            return parameters;
        }

        public override double[] GetParameterGradients()
        {
            double[] parameterGradients = new double[2 * inputDepth];

            // Copy gamma and beta gradients buffers to host
            float[] tmpGamma = new float[inputDepth];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        deltaGammaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputDepth),
                                                        tmpGamma,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            float[] tmpBeta = new float[inputDepth];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        deltaBetaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputDepth),
                                                        tmpBeta,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            // Convert to double and write into public fields
            for (int i = 0; i < inputDepth; ++i)
            {
                parameterGradients[i] = (double)tmpGamma[i];
                parameterGradients[inputDepth + i] = (double)tmpBeta[i];
            }

            return parameterGradients;
        }

        public override void SetParameters(double[] NewParameters)
        {
            // Convert to float and write into tmp arrays

            float[] tmpGamma = new float[inputDepth];
            float[] tmpBeta = new float[inputDepth];
            for (int i = 0; i < inputDepth; ++i)
            {
                tmpGamma[i] = (float)NewParameters[i];
                tmpBeta[i] = (float)NewParameters[inputDepth+i];
            }

            // Wirte arrays into buffers on device

            OpenCLSpace.ClError = Cl.EnqueueWriteBuffer(OpenCLSpace.Queue,
                                                        gammaGPU,
                                                        OpenCL.Net.Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputDepth),
                                                        tmpGamma,
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueWriteBuffer");
            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.EnqueueWriteBuffer(OpenCLSpace.Queue,
                                                        betaGPU,
                                                        OpenCL.Net.Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputDepth),
                                                        tmpBeta,
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueWriteBuffer");
            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
        }

        #endregion

    }

#endif
}