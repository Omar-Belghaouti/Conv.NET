using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace Conv.NET
{
    [Serializable]
    public class FullyConnectedLayer : Layer
    {

        #region Fields

        private double dropoutParameter;

        // Host

        private float[] weightsHost;
        private float[] biasesHost;


        // Device

        [NonSerialized]
        private Mem dropoutMaskGPU;
        
        [NonSerialized]
        private Mem weightsGPU;
        [NonSerialized]
        private Mem biasesGPU;

        [NonSerialized]
        private Mem weightsGradientsGPU;
        [NonSerialized]
        private Mem biasesGradientsGPU;

        [NonSerialized]
        private Mem weightsSpeedGPU;
        [NonSerialized]
        private Mem biasesSpeedGPU;

        // Global and local work-group sizes (for OpenCL kernels) - will be set in SetWorkGroupSizes();
        private IntPtr[] forwardGlobalWorkSizePtr;
        private IntPtr[] forwardLocalWorkSizePtr;
        private IntPtr[] backwardGlobalWorkSizePtr;
        private IntPtr[] backwardLocalWorkSizePtr;
        private IntPtr[] updateGlobalWorkSizePtr;
        private IntPtr[] updateLocalWorkSizePtr;
        private IntPtr[] constrainNormGlobalWorkSizePtr;
        private IntPtr[] constrainNormLocalWorkSizePtr;
        #endregion


        #region Properties

        public override Mem WeightsGPU
        {
            get { return weightsGPU; }
        }

        public override double DropoutParameter
        {
            set { this.dropoutParameter = value; }
        }

        #endregion


        #region Setup methods

        /// <summary>
        /// Constructor of fully connected layer type. Specify number of units as argument.
        /// </summary>
        /// <param name="nUnits"></param>
        public FullyConnectedLayer(int nUnits)
        {
            this.type = "FullyConnected";
            this.nOutputUnits = nUnits;
        }

        public override void SetupOutput()
        {
            this.outputDepth = nOutputUnits;
            this.outputHeight = 1;
            this.outputWidth = 1;

            this.outputNeurons = new Neurons(this.nOutputUnits);

#if OPENCL_ENABLED
            this.dropoutMaskGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(bool) * nOutputUnits * inputNeurons.MiniBatchSize),
                                                        out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(dropoutMaskGPU, nOutputUnits * inputNeurons.MiniBatchSize, typeof(bool));
#endif

        }

        public override void InitializeParameters(string Option)
        {

            base.InitializeParameters(Option); // makes sure this method is only call AFTER "SetupOutput()"

            if (Option == "random") // sample new parameters
            {
                //  WEIGHTS are initialized as normally distributed numbers with mean 0 and std equals to sqrt(2/nInputUnits)
                //  BIASES are initialized to a small positive number, e.g. 0.001

                this.weightsHost = new float[nOutputUnits * nInputUnits];
                this.biasesHost = new float[nOutputUnits];

                double weightsStdDev = Math.Sqrt(2.0 / (10 * nInputUnits));
                double uniformRand1;
                double uniformRand2;
                double tmp;

                for (int iRow = 0; iRow < nOutputUnits; iRow++)
                {

                    for (int iCol = 0; iCol < nInputUnits; iCol++)
                    {
                        uniformRand1 = Global.rng.NextDouble();
                        uniformRand2 = Global.rng.NextDouble();
                        // Use a Box-Muller transform to get a random normal(0,1)
                        tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);
                        tmp = weightsStdDev * tmp; // rescale

                        weightsHost[iRow * nInputUnits + iCol] = (float)tmp;
                    }
                    biasesHost[iRow] = 0.00f;
                }
            }
            // else Option must be ''load'' => do not sample parameters, just load them from host to device

            int weightBufferSize = sizeof(float) * (outputNeurons.NumberOfUnits * inputNeurons.NumberOfUnits);
            int biasesBufferSize = sizeof(float) * outputNeurons.NumberOfUnits;

            this.weightsGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                    (IntPtr)weightBufferSize,
                                                    weightsHost,
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.CreateBuffer");

            this.biasesGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                    (IntPtr)biasesBufferSize,
                                                    biasesHost,
                                                    out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.CreateBuffer");

            // Also create weightsGradients and biasesGradients buffers and initialize them to zero

            this.weightsGradientsGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)weightBufferSize,
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(weightsGradientsGPU, (nInputUnits * nOutputUnits), typeof(float));

            this.biasesGradientsGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)biasesBufferSize,
                                                            out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(biasesGradientsGPU, nOutputUnits, typeof(float));

            // Also create weightsSpeed and biasesSpeed buffers and initialize them to zero

            this.weightsSpeedGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)weightBufferSize,
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(weightsSpeedGPU, (nInputUnits * nOutputUnits), typeof(float));

            this.biasesSpeedGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)biasesBufferSize,
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.CreateBuffer");
            OpenCLSpace.WipeBuffer(biasesSpeedGPU, nOutputUnits, typeof(float));

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

            int miniBatchSize = outputNeurons.MiniBatchSize;


            // FeedForward (2D) ________________________________________________________________________________
            
            // Local
            int optimalToBaseRatio = OpenCLSpace.OPTIMAL_GROUP_SIZE / OpenCLSpace.BASE_GROUP_SIZE;
            this.forwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.BASE_GROUP_SIZE, (IntPtr)optimalToBaseRatio };

            // Global
            int smallestMultiple0 = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(nOutputUnits) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            int smallestMultiple1 = (int)(optimalToBaseRatio * Math.Ceiling((double)(miniBatchSize) / (double)optimalToBaseRatio));
            this.forwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple0, (IntPtr)smallestMultiple1 };


            // BackPropagate (2D) _________________________________________________________________________________

            // Local
            this.backwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.BASE_GROUP_SIZE, (IntPtr)optimalToBaseRatio };
            
            // Global
            smallestMultiple0 = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(nInputUnits) / (double)OpenCLSpace.BASE_GROUP_SIZE)); // input this time!
            this.backwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple0, (IntPtr)smallestMultiple1 };


            // UpdateSpeeds and UpdateParameters (2D) ________________________________________________________________

            // Local
            this.updateLocalWorkSizePtr = new IntPtr[] { (IntPtr)optimalToBaseRatio, (IntPtr)OpenCLSpace.BASE_GROUP_SIZE }; // product is OPTIMAL_WORK_SIZE

            // Global
            smallestMultiple0 = (int)(optimalToBaseRatio * Math.Ceiling((double)(nOutputUnits) / (double)optimalToBaseRatio));
            smallestMultiple1 = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(nInputUnits) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.updateGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple0, (IntPtr)smallestMultiple1 };


            // Max norm constrain
            this.constrainNormLocalWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.BASE_GROUP_SIZE };
            int smallestMultipleAux = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(nOutputUnits) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.constrainNormGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultipleAux };

        }


        public override void CopyBuffersToHost()
        {
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                        weightsGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nInputUnits * nOutputUnits),
                                                        weightsHost,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer weightsGPU");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.EnqueueReadBuffer( OpenCLSpace.Queue,
                                                        biasesGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nOutputUnits),
                                                        biasesHost,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer biasesGPU");

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
            Utils.FCForwardTimer.Start();
#endif

#if OPENCL_ENABLED
            // Set kernel arguments
            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.FCForward, 0, outputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 1, inputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 2, weightsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 3, biasesGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 4, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 5, (IntPtr)sizeof(int), nOutputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 6, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 7, (IntPtr)sizeof(float), (float)dropoutParameter);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 8, (IntPtr)sizeof(ulong), (ulong)Guid.NewGuid().GetHashCode()); // this should be quite a good random seed
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 9, dropoutMaskGPU);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.FeedForward(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.FCForward,
                                                            2,
                                                            null,
                                                            forwardGlobalWorkSizePtr,
                                                            forwardLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.FeedForward(): Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            // TODO: add dropout CPU
            // Generate dropout mask
            if (dropoutParameter < 1)
            {
                for (int iUnit = 0; iUnit < nOutputUnits * inputNeurons.MiniBatchSize; ++iUnit)
                    dropoutMask[iUnit] = Global.RandomDouble() < dropoutParameter;
            }

            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                double[] unbiasedOutput = Utils.MultiplyMatrixByVector(weights, inputNeurons.GetHost()[m]);
                this.outputNeurons.SetHost(m, unbiasedOutput.Zip(biases, (x, y) => x + y).ToArray());
            }
#endif


#if TIMING_LAYERS
            Utils.FCForwardTimer.Stop();
#endif
        }

        public override void BackPropagate()
        {
#if TIMING_LAYERS
            Utils.FCBackpropTimer.Start();
#endif

#if OPENCL_ENABLED

            // Set kernel arguments
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 0, inputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 1, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 2, weightsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 3, dropoutMaskGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 4, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 5, (IntPtr)sizeof(int), nOutputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 6, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.BackPropagate(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.FCBackward,
                                                            2,
                                                            null,
                                                            backwardGlobalWorkSizePtr,
                                                            backwardLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.BackPropagate(): Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {

                inputNeurons.DeltaHost[m] = Utils.MultiplyMatrixTranspByVector(weights, outputNeurons.DeltaHost[m]);
            }
#endif

#if TIMING_LAYERS
            Utils.FCBackpropTimer.Stop();
#endif
        }


        public override void UpdateSpeeds(double learningRate, double momentumCoefficient, double weightDecayCoefficient)
        {
#if TIMING_LAYERS
            Utils.FCUpdateSpeedsTimer.Start();
#endif

#if DEBUGGING_STEPBYSTEP_FC
            float[,] weightsBeforeUpdate = new float[output.NumberOfUnits, input.NumberOfUnits];
            /* ------------------------- DEBUGGING --------------------------------------------- */
#if OPENCL_ENABLED
            // Display weights before update
            
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                            weightsGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * input.NumberOfUnits * sizeof(float)),
                                            weightsBeforeUpdate,  // destination
                                            0,
                                            null,
                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer weightsBeforeUpdate");
#else
            weightsBeforeUpdate = weights;
#endif
            Console.WriteLine("\nWeights BEFORE update:");
            for (int i = 0; i < weightsBeforeUpdate.GetLength(0); i++)
            {
                for (int j = 0; j < weightsBeforeUpdate.GetLength(1); j++)
                    Console.Write("{0}  ", weightsBeforeUpdate[i, j]);
                Console.WriteLine();
            }
            Console.WriteLine();
            Console.ReadKey();
            

            // Display biases before update
            float[] biasesBeforeUpdate = new float[output.NumberOfUnits];
#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                            biasesGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * sizeof(float)),
                                            biasesBeforeUpdate,  // destination
                                            0,
                                            null,
                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer biasesBeforeUpdate");
#else
            biasesBeforeUpdate = biases;
#endif
            Console.WriteLine("\nBiases BEFORE update:");
            for (int i = 0; i < biasesBeforeUpdate.Length; i++)
            {
                Console.Write("{0}  ", biasesBeforeUpdate[i]);
            }
            Console.WriteLine();
            Console.ReadKey();
            

            // Display weight update speed before update
            
            float[,] tmpWeightsUpdateSpeed = new float[output.NumberOfUnits, input.NumberOfUnits];
#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                            weightsUpdateSpeedGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * input.NumberOfUnits * sizeof(float)),
                                            tmpWeightsUpdateSpeed,  // destination
                                            0,
                                            null,
                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer weightsUpdateSpeed");
#else
            tmpWeightsUpdateSpeed = weightsUpdateSpeed;
#endif
            Console.WriteLine("\nWeight update speed BEFORE update:");
            for (int i = 0; i < tmpWeightsUpdateSpeed.GetLength(0); i++)
            {
                for (int j = 0; j < tmpWeightsUpdateSpeed.GetLength(1); j++)
                    Console.Write("{0}  ", tmpWeightsUpdateSpeed[i, j]);
                Console.WriteLine();
            }
            Console.WriteLine();
            Console.ReadKey();

            // Display input activations before update

            /*
            float[] inputActivations = new float[input.NumberOfUnits];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                            input.ActivationsGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(input.NumberOfUnits * sizeof(float)),
                                            inputActivations,  // destination
                                            0,
                                            null,
                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer inputActivations");

            Console.WriteLine("\nInput activations BEFORE update:");

            for (int j = 0; j < inputActivations.Length; j++)
            {
                Console.Write("{0}  ", inputActivations[j]);
            }
            Console.WriteLine();
            Console.ReadKey();
            


            // Display output delta before update

            float[] outputDelta = new float[output.NumberOfUnits];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                            output.DeltaGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * sizeof(float)),
                                            outputDelta,  // destination
                                            0,
                                            null,
                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer outputDelta");

            Console.WriteLine("\nOutput delta BEFORE update:");

            for (int i = 0; i < outputDelta.Length; i++)
            {
                Console.Write("{0}", outputDelta[i]);
                Console.WriteLine();
            }
            Console.WriteLine();
            Console.ReadKey();
            */



            /*------------------------- END DEBUGGING --------------------------------------------- */
#endif

#if OPENCL_ENABLED
            // Set kernel arguments
            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 0, weightsSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 1, biasesSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 2, weightsGradientsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 3, biasesGradientsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 4, inputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 5, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 6, dropoutMaskGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 7, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 8, (IntPtr)sizeof(int), nOutputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 9, (IntPtr)sizeof(float), (float)momentumCoefficient);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 10, (IntPtr)sizeof(float), (float)learningRate);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 11, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 12, weightsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 13, (IntPtr)sizeof(float), (float)weightDecayCoefficient);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.UpdateSpeeds(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.FCUpdateSpeeds,
                                                            2,
                                                            null,
                                                            updateGlobalWorkSizePtr,
                                                            updateLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.UpdateSpeeds(): Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

#else
            int miniBatchSize = inputNeurons.MiniBatchSize;

            for (int m = 0; m < miniBatchSize; m++)
            {
                for (int i = 0; i < nOutputUnits; i++)
                {
                    // weights speed

                    for (int j = 0; j < nInputUnits; j++)
                    {
                        if (m == 0)
                            weightsUpdateSpeed[i, j] *= momentumCoefficient;

                        weightsUpdateSpeed[i, j] -= learningRate/miniBatchSize * inputNeurons.GetHost()[m][j] * outputNeurons.DeltaHost[m][i];
#if GRADIENT_CHECK
                        weightsGradients[i, j] = inputNeurons.GetHost()[m][j] * outputNeurons.DeltaHost[m][i];
#endif
                    
                    }

                    // update biases
                    if (m == 0)
                            biasesUpdateSpeed[i] *= momentumCoefficient;

                    biasesUpdateSpeed[i] -= learningRate/miniBatchSize * outputNeurons.DeltaHost[m][i];

#if GRADIENT_CHECK
                    biasesGradients[i] = outputNeurons.DeltaHost[m][i];
#endif

                }

            } // end loop over mini-batch
#endif


#if TIMING_LAYERS
            Utils.FCUpdateSpeedsTimer.Stop();
#endif

        }

        public override void UpdateParameters(double weightMaxNorm)
        {

#if TIMING_LAYERS
            Utils.FCUpdateParametersTimer.Start();
#endif

#if OPENCL_ENABLED
            // Set kernel arguments
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 0, weightsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 1, biasesGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 2, weightsSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 3, biasesSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 4, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 5, (IntPtr)sizeof(int), nOutputUnits);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.UpdateParameters(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.FCUpdateParameters,
                                                            2,
                                                            null,
                                                            updateGlobalWorkSizePtr,
                                                            updateLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.UpdateParameters(): Cl.EnqueueNDRangeKernel");


            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");


            // Now constrain norm of each weight vector
            if (!double.IsInfinity(weightMaxNorm))
            {
                // Set kernel arguments
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.FCConstrainWeightNorm, 0, weightsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCConstrainWeightNorm, 1, (IntPtr)sizeof(int), nOutputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCConstrainWeightNorm, 2, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCConstrainWeightNorm, 3, (IntPtr)sizeof(float), (float)weightMaxNorm);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FCConstrainWeightNorm(): Cl.SetKernelArg");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                                OpenCLSpace.FCConstrainWeightNorm,
                                                                1,
                                                                null,
                                                                constrainNormGlobalWorkSizePtr,
                                                                constrainNormLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FCConstrainWeightNorm(): Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
            }
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
                

                for (int i = 0; i < nOutputUnits; i++)
                {
                    // weights update

                    for (int j = 0; j < nInputUnits; j++)
                    {
                        weights[i, j] += weightsUpdateSpeed[i, j];
                    }

                    // update biases
                    biases[i] += biasesUpdateSpeed[i];
                }
#endif

#if TIMING_LAYERS
                Utils.FCUpdateParametersTimer.Stop();
#endif
        }
        
        #endregion


        #region Gradient check

        public override double[] GetParameters()
        {
            int nParameters = nInputUnits * nOutputUnits + nOutputUnits;
            double[] parameters = new double[nParameters];

            // Copy weights and biases buffers to host
            float[] tmpWeights = new float[nInputUnits * nOutputUnits];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        weightsGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nInputUnits * nOutputUnits),
                                                        tmpWeights,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            float[] tmpBiases = new float[nOutputUnits];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        biasesGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nOutputUnits),
                                                        tmpBiases,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            // Convert to double and write into parameters array
            for (int i = 0; i < nInputUnits*nOutputUnits; ++i)
            {
                parameters[i] = (double)tmpWeights[i];
            }
            for (int i = 0; i < nOutputUnits; ++i)
            {
                parameters[nInputUnits * nOutputUnits + i] = (double)tmpBiases[i];
            }

            return parameters;
        }

        public override double[] GetParameterGradients()
        {
            int nParameters = nInputUnits * nOutputUnits + nOutputUnits;
            double[] parameterGradients = new double[nParameters];

            // Copy weights and biases gradients buffers to host
            float[] tmpWeightsGrad = new float[nInputUnits * nOutputUnits];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        weightsGradientsGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nInputUnits * nOutputUnits),
                                                        tmpWeightsGrad,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            float[] tmpBiasesGrad = new float[nOutputUnits];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        biasesGradientsGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nOutputUnits),
                                                        tmpBiasesGrad,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            // Convert to double and write into parameterGradients
            //Console.WriteLine("Weight gradients:\n");
            for (int i = 0; i < nInputUnits * nOutputUnits; ++i)
            {
                parameterGradients[i] = (double)tmpWeightsGrad[i];
                //Console.Write(" {0}", tmpWeightsGrad[i]);
            }
            //Console.ReadKey();
            for (int i = 0; i < nOutputUnits; ++i)
            {
                parameterGradients[nInputUnits * nOutputUnits + i] = (double)tmpBiasesGrad[i];
            }

            return parameterGradients;
        }

        public override void SetParameters(double[] NewParameters)
        {
            // Convert to float and write into tmp arrays

            float[] tmpWeights = new float[nInputUnits * nOutputUnits];
            float[] tmpBiases = new float[nOutputUnits];
            for (int i = 0; i < nInputUnits * nOutputUnits; ++i)
            {
                tmpWeights[i] = (float)NewParameters[i];
            }
            for (int i = 0; i < nOutputUnits; ++i)
            {
                tmpBiases[i] = (float)NewParameters[nInputUnits * nOutputUnits + i];
            }

            // Write arrays into buffers on device

            OpenCLSpace.ClError = Cl.EnqueueWriteBuffer(OpenCLSpace.Queue,
                                                        weightsGPU,
                                                        OpenCL.Net.Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nInputUnits * nOutputUnits),
                                                        tmpWeights,
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueWriteBuffer");
            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.EnqueueWriteBuffer(OpenCLSpace.Queue,
                                                        biasesGPU,
                                                        OpenCL.Net.Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * nOutputUnits),
                                                        tmpBiases,
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
}
