using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCL.Net;

namespace JaNet
{
#if OPENCL_ENABLED

    class BatchNormFC : Layer
    {

        #region Fields

        private bool isEpochBeginning;
        private bool isTraining;
        private bool isPreInference;
        private bool isInference;

        private int iCumulativeAverage;

        // OpenCL-only fields

        private Mem meanGPU;
        private Mem varianceGPU;

        private Mem cumulativeMeanGPU;
        private Mem cumulativeVarianceGPU;

        private Mem normalizedInputGPU;

        private Mem gammaGPU;
        private Mem betaGPU;

        private Mem deltaGammaGPU;
        private Mem deltaBetaGPU;

        private Mem gammaSpeedGPU;
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
        }

        public override void InitializeParameters()
        {
            this.iCumulativeAverage = 0;
            this.isEpochBeginning = true;

            this.isTraining = true;
            this.isPreInference = false;
            this.isInference = false;

            // 1. Initialize OpenCL buffers for mean, variance and their cumulative averages
          
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

            // (initialize cumulative means to zero...)

            this.cumulativeMeanGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * nInputUnits),
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(cumulativeMeanGPU, nInputUnits, typeof(float));

            // (...and variances to one)
            float[] ones = new float[nInputUnits];
            for (int i = 0; i < nInputUnits; ++i)
                ones[i] = 1.0f;
            this.cumulativeVarianceGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                                (IntPtr)(sizeof(float) * nInputUnits),
                                                                ones,
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(cumulativeVarianceGPU, nInputUnits, typeof(float));
            

            // 2. Initialize OpenCL buffers for normalized input values (needed for backprop)
            
            this.normalizedInputGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * nInputUnits * inputNeurons.MiniBatchSize),
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(normalizedInputGPU, nInputUnits * inputNeurons.MiniBatchSize, typeof(float));


            // 3. Initialize OpenCL buffers for learnable parameters gamma and beta, their gradients, and their update speed.
            // Write ones in gammas and zeros in betas (identity function in the beginning). Write zeros in speeds.

            this.gammaGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                    (IntPtr)(sizeof(float) * nInputUnits),
                                                    ones,
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            float[] zeros = new float[nInputUnits];
            this.betaGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                (IntPtr)(sizeof(float) * nInputUnits),
                                                zeros,
                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            
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

            this.gammaSpeedGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                        MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                        (IntPtr)(sizeof(float) * nInputUnits),
                                                        zeros,
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.betaSpeedGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                        MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                        (IntPtr)(sizeof(float) * nInputUnits),
                                                        zeros,
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
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

                // Wipe cumulative means and variances
                OpenCLSpace.WipeBuffer(cumulativeMeanGPU, nInputUnits, typeof(float));
                OpenCLSpace.WipeBuffer(cumulativeVarianceGPU, nInputUnits, typeof(float));

                isEpochBeginning = false;
            }


            // If training, compute means and variances, and update cumulative averages
            if (isTraining || isPreInference)
            {
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 0, meanGPU);
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
            if (isTraining || isPreInference)
            {
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 3, meanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCForward, 4, varianceGPU);
            }
            else if (isInference)
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
                                                            nParametersGlobalWorkSizePtr,
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

    }

#endif
}