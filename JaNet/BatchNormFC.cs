using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCL.Net;

namespace JaNet
{
#if OPENCL_ENABLED

    [Serializable]
    class BatchNormFC : Layer
    {

        #region Fields

        private bool isEpochBeginning;
        private bool isTraining;
        private bool isPreInference;
        private bool isInference;

        private int iCumulativeAverage;

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
        private Mem deltaGammaGPU;
        [NonSerialized]
        private Mem deltaBetaGPU;

        [NonSerialized]
        private Mem gammaSpeedGPU;
        [NonSerialized]
        private Mem betaSpeedGPU;


        // Work group sizes
        private IntPtr[] optimalLocalWorkSizePtr;

        private IntPtr[] nUnitsGlobalWorkSizePtr;
        private IntPtr[] nParametersGlobalWorkSizePtr;
        private IntPtr[] nActivationsGlobalWorkSizePtr;

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
        /// Constructor of BatchNormLayer.
        /// </summary>
        public BatchNormFC()
        {
            this.type = "BatchNormFC";
        }

        public override void SetupOutput()
        {
            this.outputWidth = inputWidth;
            this.outputHeight = inputHeight;
            this.outputDepth = inputDepth;

            this.nOutputUnits = nInputUnits;
            this.outputNeurons = new Neurons(nOutputUnits);

            // Also initialize OpenCL buffers for mean, variance, their cumulative averages, and normalized input activations
            
            this.meanGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                MemFlags.ReadWrite,
                                                (IntPtr)(sizeof(float) * nInputUnits),
                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(meanGPU, nInputUnits, typeof(float));

            this.varianceGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                    MemFlags.ReadWrite,
                                                    (IntPtr)(sizeof(float) * nInputUnits),
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(varianceGPU, nInputUnits, typeof(float));

            this.cumulativeMeanGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * nInputUnits),
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(cumulativeMeanGPU, nInputUnits, typeof(float));

            this.cumulativeVarianceGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)(sizeof(float) * nInputUnits),
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(cumulativeVarianceGPU, nInputUnits, typeof(float));
            
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

            if (Option == "random") // initialize parameters on host
            {
                // Gamma parameters are initialized to one
                gammaHost = new float[nInputUnits];
                for (int i = 0; i < nInputUnits; ++i)
                    gammaHost[i] = 1.0f;
                // And beta parameters to zero
                betaHost = new float[nInputUnits];
            }
            // else Option must be ''load'' => do not initialized parameters, just load them from host to device

            // Tranfer parameters to device

            this.gammaGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                    (IntPtr)(sizeof(float) * nInputUnits),
                                                    gammaHost,
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.betaGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                (IntPtr)(sizeof(float) * nInputUnits),
                                                betaHost,
                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            
            // Also create buffers for parameter gradients

            this.deltaGammaGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * nInputUnits),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(deltaGammaGPU, nInputUnits, typeof(float));

            this.deltaBetaGPU = (Mem)Cl.CreateBuffer(   OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * nInputUnits),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(deltaBetaGPU, nInputUnits, typeof(float));

            // And for parameter update speed

            this.gammaSpeedGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * nInputUnits),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(gammaSpeedGPU, nInputUnits, typeof(float));

            this.betaSpeedGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * nInputUnits),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(betaSpeedGPU, nInputUnits, typeof(float));
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
            this.optimalLocalWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.OPTIMAL_GROUP_SIZE };

            // _____________________________________________________________________________________________________________________
            // Global 1D worksize of size nInputUnits: Use for ComputeMeansVariances() and UpdateParameters() 

            int smallestMultiple = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE * Math.Ceiling((double)(nInputUnits) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
            this.nUnitsGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple };

            // _____________________________________________________________________________________________________________________
            // Global 1D worksize of size 2*nInputUnits: Use for UpdateSpeeds()
            this.nParametersGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(2 * smallestMultiple) };

            // _____________________________________________________________________________________________________________________
            // Global 1D worksize equal to the total number of activations. Use for FeedForward() and BackPropagate() kernels 

            // Global
            smallestMultiple = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE * Math.Ceiling((double)(nInputUnits * inputNeurons.MiniBatchSize) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
            this.nActivationsGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple };

        }

        #endregion


        #region Methods

        public override void FeedForward()
        {
#if TIMING_LAYERS
            Utils.BNFCForwardTimer.Start();
#endif
            if (isEpochBeginning)
            {
                iCumulativeAverage = 0;

                // Wipe cumulative means and variances (theoretically, this is redundant)
                OpenCLSpace.WipeBuffer(cumulativeMeanGPU, nInputUnits, typeof(float));
                OpenCLSpace.WipeBuffer(cumulativeVarianceGPU, nInputUnits, typeof(float));

                isEpochBeginning = false;
            }


            // If training, compute means and variances, and update cumulative averages
            if (isTraining || isPreInference)
            {
                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 0, meanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 1, varianceGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 2, cumulativeMeanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 3, cumulativeVarianceGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 4, inputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 5, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 6, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 7, (IntPtr)sizeof(int), Convert.ToInt32(isPreInference));
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 8, (IntPtr)sizeof(int), iCumulativeAverage);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                                OpenCLSpace.BNFCComputeMeansVariances,
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

                if (isPreInference)
                    iCumulativeAverage++; // increase cumulative average counter
            }

            
            //Normalize input, scale and shift

            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.BNFCForward, 0, outputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 1, normalizedInputGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 2, inputNeurons.ActivationsGPU);
            if (isTraining)
            {
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 3, meanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 4, varianceGPU);
            }
            else if (isPreInference || isInference)
            {
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 3, cumulativeMeanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 4, cumulativeVarianceGPU);
            }
            else
                throw new InvalidOperationException("ERROR: BatchNormConv is currently not in training mode, nor pre-inference, nor inference.");
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 5, gammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 6, betaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 7, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 8, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                            OpenCLSpace.BNFCForward,
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
            Utils.BNFCForwardTimer.Stop();
#endif

        }


        public override void BackPropagate()
        {

#if TIMING_LAYERS
            Utils.BNFCBackpropTimer.Start();
#endif

            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 0, inputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 1, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 2, normalizedInputGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 3, gammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 4, varianceGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 5, deltaGammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 6, deltaBetaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 7, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 8, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                            OpenCLSpace.BNFCBackPropagate,
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
            Utils.BNFCBackpropTimer.Stop();
#endif
        }

        public override void UpdateSpeeds(double learningRate, double momentumMultiplier)
        {
#if TIMING_LAYERS
            Utils.BNFCUpdateSpeedsTimer.Stop();
#endif
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 0, gammaSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 1, betaSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 2, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 3, normalizedInputGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 4, deltaGammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 5, deltaBetaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 6, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 7, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 8, (IntPtr)sizeof(float), (float)momentumMultiplier);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 9, (IntPtr)sizeof(float), (float)learningRate);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                            OpenCLSpace.BNFCUpdateSpeeds,
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

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");


            /* ------------------------- DEBUGGING ---------------------------------------------

            // Display gamma gradient
            
            float[] deltaGgamma = new float[nInputUnits];

            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        deltaGammaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(nInputUnits * sizeof(float)),
                                                        deltaGgamma,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

            Console.WriteLine("\nGradient wrt gamma:\n");
            for (int i = 0; i < nInputUnits; i++)
                Console.Write("{0}  ", deltaGgamma[i]);
            //Console.ReadKey();
            
            // Display beta
            float[] deltaBeta = new float[nInputUnits];

            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        deltaBetaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(nInputUnits * sizeof(float)),
                                                        deltaBeta,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.clEnqueueReadBuffer");

            Console.WriteLine("\n\nGradient wrt beta:\n");
            for (int i = 0; i < nInputUnits; i++)
                Console.Write("{0}  ", deltaBeta[i]);
            Console.ReadKey();


            /* ------------------------- END DEBUGGING --------------------------------------------- */


#if TIMING_LAYERS
            Utils.BNFCUpdateSpeedsTimer.Stop();
#endif
        }


        public override void UpdateParameters(double weightDecayCoeff)
        {

#if TIMING_LAYERS
            Utils.BNFCUpdateParametersTimer.Start();
#endif

            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.BNFCUpdateParameters, 0, gammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateParameters, 1, betaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateParameters, 2, gammaSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateParameters, 3, betaSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateParameters, 4, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                            OpenCLSpace.BNFCUpdateParameters,
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

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");


            /* ------------------------- DEBUGGING ---------------------------------------------

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

            Console.WriteLine("\n\nUpdated gammas are:\n");
            for (int i = 0; i < nInputUnits; i++)
                Console.Write("{0}  ", gamma[i]);
            //Console.ReadKey();

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

#if TIMING_LAYERS
            Utils.BNFCUpdateParametersTimer.Stop();
#endif

        }

        #endregion


        #region Gradient check

        public override double[] GetParameters()
        {
            double[] parameters = new double[2 * nInputUnits];

            // Copy gamma and beta buffers to host
            float[] tmpGamma = new float[nInputUnits];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        gammaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nInputUnits),
                                                        tmpGamma,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            float[] tmpBeta = new float[nInputUnits];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        betaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nInputUnits),
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
            for (int i = 0; i < nInputUnits; ++i)
            {
                parameters[i] = (double)tmpGamma[i];
                parameters[nInputUnits + i] = (double)tmpBeta[i];
            }

            return parameters;
        }

        public override double[] GetParameterGradients()
        {
            double[] parameterGradients = new double[2 * nInputUnits];

            // Copy gamma and beta gradients buffers to host
            float[] tmpGammaGrad = new float[nInputUnits];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        deltaGammaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nInputUnits),
                                                        tmpGammaGrad,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            float[] tmpBetaGrad = new float[nInputUnits];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        deltaBetaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nInputUnits),
                                                        tmpBetaGrad,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            // Convert to double and write into public fields
            for (int i = 0; i < nInputUnits; ++i)
            {
                parameterGradients[i] = (double)tmpGammaGrad[i];
                parameterGradients[nInputUnits + i] = (double)tmpBetaGrad[i];
            }

            return parameterGradients;
        }

        public override void SetParameters(double[] NewParameters)
        {
            // Convert to float and write into tmp arrays

            float[] tmpGamma = new float[nInputUnits];
            float[] tmpBeta = new float[nInputUnits];
            for (int i = 0; i < nInputUnits; ++i)
            {
                tmpGamma[i] = (float)NewParameters[i];
                tmpBeta[i] = (float)NewParameters[nInputUnits + i];
            }

            // Write arrays into buffers on device

            OpenCLSpace.ClError = Cl.EnqueueWriteBuffer(OpenCLSpace.Queue,
                                                        gammaGPU,
                                                        OpenCL.Net.Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nInputUnits),
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
                                                        (IntPtr)(sizeof(float) * nInputUnits),
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


        public override double[] GetInput()
        {
            int inputArraySize = nInputUnits * inputNeurons.MiniBatchSize;
            double[] input = new double[inputArraySize];

            // Copy device buffer to host
            float[] tmpInput = new float[inputArraySize];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                        inputNeurons.ActivationsGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputArraySize),
                                                        tmpInput,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            // Convert to double and write into public fields
            for (int i = 0; i < inputArraySize; ++i)
            {
                input[i] = (double)tmpInput[i];
            }

            return input;
        }

        public override double[] GetInputGradients()
        {
            int inputArraySize = nInputUnits * inputNeurons.MiniBatchSize;
            double[] inputGradients = new double[inputArraySize];

            // Copy device buffer to host
            float[] tmpInputGradients = new float[inputArraySize];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        inputNeurons.DeltaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputArraySize),
                                                        tmpInputGradients,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            // Convert to double and write into public fields
            for (int i = 0; i < inputArraySize; ++i)
            {
                inputGradients[i] = (double)tmpInputGradients[i];
            }

            return inputGradients;
        }

        public override void SetInput(double[] NewInput )
        {
            // Convert to float and write into tmp arrays
            int inputArraySize = nInputUnits * inputNeurons.MiniBatchSize;
            float[] tmpInput = new float[inputArraySize];
            for (int i = 0; i < inputArraySize; ++i)
                tmpInput[i] = (float)NewInput[i];

            // Write arrays into buffers on device

            OpenCLSpace.ClError = Cl.EnqueueWriteBuffer(OpenCLSpace.Queue,
                                                        inputNeurons.ActivationsGPU,
                                                        OpenCL.Net.Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputArraySize),
                                                        tmpInput,
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