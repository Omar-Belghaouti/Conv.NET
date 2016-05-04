using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using OpenCL.Net;

namespace JaNet
{
#if OPENCL_ENABLED

    class BatchNormConv : Layer
    {

        #region Fields

        private bool isEpochBeginning;
        private bool isTraining;
        private int iCumulativeAverage;

        // OpenCL-only fields

        private Mem meanGPU;
        private Mem varianceGPU;

        private Mem cumulativeMeanGPU;
        private Mem cumulativeVarianceGPU;

        private Mem deltaMeanGPU;
        private Mem deltaVarianceGPU;

        private Mem normalizedInputGPU;

        private Mem betaGPU;
        private Mem gammaGPU;

        private Mem betaSpeedGPU;
        private Mem gammaSpeedGPU;

        // Work group sizes
        private IntPtr[] nUnitsGlobalWorkSizePtr;
        private IntPtr[] nUnitsLocalWorkSizePtr;

        private IntPtr[] nParametersGlobalWorkSizePtr;

        private IntPtr[] nActivationsGlobalWorkSizePtr;
        private IntPtr[] nActivationsLocalWorkSizePtr;

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
        /// Constructor of BatchNormLayer.
        /// </summary>
        public BatchNormConv()
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

            // Initialize OpenCL buffers
            int bufferSize = sizeof(float) * nInputUnits;

            this.meanGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                MemFlags.ReadWrite,
                                                (IntPtr)bufferSize,
                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.varianceGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                    MemFlags.ReadWrite,
                                                    (IntPtr)bufferSize,
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.cumulativeMeanGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)bufferSize,
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.cumulativeVarianceGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)bufferSize,
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.deltaMeanGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)bufferSize,
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.deltaVarianceGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)bufferSize,
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            // Write zeros, just in case
            OpenCLSpace.WipeBuffer(meanGPU, inputDepth, typeof(float));
            OpenCLSpace.WipeBuffer(varianceGPU, inputDepth, typeof(float));
            OpenCLSpace.WipeBuffer(cumulativeMeanGPU, inputDepth, typeof(float));
            OpenCLSpace.WipeBuffer(cumulativeVarianceGPU, inputDepth, typeof(float));
            OpenCLSpace.WipeBuffer(deltaMeanGPU, inputDepth, typeof(float));
            OpenCLSpace.WipeBuffer(deltaVarianceGPU, inputDepth, typeof(float));

            // Initialize OpenCL buffers for normalized input values (needed for backprop)
            this.normalizedInputGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * nInputUnits * inputNeurons.MiniBatchSize),
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(normalizedInputGPU, nInputUnits * inputNeurons.MiniBatchSize, typeof(float));


            // Initialize OpenCL buffers for learnable parameters and their update speed.
            // Write ones in gammas and zeros in betas (identity function in the beginning). Write zeros in speeds.


            float[] ones = new float[nInputUnits];
            for (int i = 0; i < nInputUnits; ++i)
                ones[i] = 1.0f;
            this.gammaGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                    (IntPtr)bufferSize,
                                                    ones,
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            float[] zeros = new float[nInputUnits];
            this.betaGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                (IntPtr)bufferSize,
                                                zeros,
                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.gammaSpeedGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                        MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                        (IntPtr)bufferSize,
                                                        zeros,
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            this.betaSpeedGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                        MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                        (IntPtr)bufferSize,
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

            // _____________________________________________________________________________________________________________________
            // 1D worksize of size nInputUnits: Use for ComputeMeansVariances(), UpdateSpeeds() and UpdateParameters() 

            // Local
            this.nUnitsLocalWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.OPTIMAL_GROUP_SIZE };

            // Global
            int smallestMultiple = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE * Math.Ceiling((double)(nInputUnits) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
            this.nUnitsGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple };
            this.nParametersGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(2 * smallestMultiple) }; // use for gradients kernel

            // _____________________________________________________________________________________________________________________
            // 1D worksize equal to the total number of activations. Use for FeedForward() and BackPropagate() kernels 

            // Local
            this.nActivationsLocalWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.OPTIMAL_GROUP_SIZE };

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
                isEpochBeginning = false;
            }

            // If training, compute means and variances, and update cumulative averages
            if (isTraining)
            {
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 0, meanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 1, varianceGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 2, cumulativeMeanGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 3, cumulativeVarianceGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 4, inputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 5, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 6, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCComputeMeansVariances, 7, (IntPtr)sizeof(int), iCumulativeAverage);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                                OpenCLSpace.BNFCComputeMeansVariances,
                                                                1,
                                                                null,
                                                                nUnitsGlobalWorkSizePtr,
                                                                nUnitsLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

                // increase average counter
                iCumulativeAverage++;
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

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
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

#if TIMING_LAYERS
            Utils.BNFCForwardTimer.Start();
#endif

        }


        public override void BackPropagate()
        {

#if TIMING_LAYERS
            Utils.BNFCBackpropTimer.Start();
#endif
            //Compute gradients of loss function wrt mean and variance in each feature map

            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 0, deltaMeanGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 1, deltaVarianceGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 2, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 3, normalizedInputGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 4, gammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 5, varianceGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 6, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCGradientMeanVariance, 7, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                            OpenCLSpace.BNFCGradientMeanVariance,
                                                            1,
                                                            null,
                                                            nParametersGlobalWorkSizePtr,
                                                            nUnitsLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            //Backpropagate to input deltas using above gradients as auxiliary variables

            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 0, inputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 1, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 2, inputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 3, gammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 4, meanGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 5, varianceGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 6, deltaMeanGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 7, deltaVarianceGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 8, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCBackPropagate, 9, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
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
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 4, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 5, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 6, (IntPtr)sizeof(float), (float)momentumMultiplier);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNFCUpdateSpeeds, 7, (IntPtr)sizeof(float), (float)learningRate);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                            OpenCLSpace.BNFCUpdateSpeeds,
                                                            1,
                                                            null,
                                                            nParametersGlobalWorkSizePtr,
                                                            nUnitsLocalWorkSizePtr,
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

            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.BNUpdateParameters, 0, gammaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNUpdateParameters, 1, betaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNUpdateParameters, 2, gammaSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNUpdateParameters, 3, betaSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.BNUpdateParameters, 4, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                            OpenCLSpace.BNUpdateParameters,
                                                            1,
                                                            null,
                                                            nUnitsGlobalWorkSizePtr,
                                                            nUnitsLocalWorkSizePtr,
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

#if TIMING_LAYERS
            Utils.BNFCUpdateParametersTimer.Stop();
#endif

        }

        #endregion

    }

#endif
}