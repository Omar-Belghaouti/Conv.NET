using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    class FullyConnectedLayer : Layer
    {

        #region Fields

        private double dropoutParameter;
        

#if OPENCL_ENABLED

        private Mem weightsGPU;
        private Mem biasesGPU;

        private Mem weightsSpeedGPU;
        private Mem biasesSpeedGPU;

        // Global and local work-group sizes (for OpenCL kernels) - will be set in SetWorkGroupSizes();
        private IntPtr[] forwardGlobalWorkSizePtr;
        private IntPtr[] forwardLocalWorkSizePtr;
        private IntPtr[] backwardGlobalWorkSizePtr;
        private IntPtr[] backwardLocalWorkSizePtr;
        private IntPtr[] updateGlobalWorkSizePtr;
        private IntPtr[] updateLocalWorkSizePtr;
#else
        // Host
        private double[,] weights;
        private double[] biases;

        private double[,] weightsUpdateSpeed;
        private double[] biasesUpdateSpeed;

        private bool[] dropoutMask;

#if GRADIENT_CHECK
        private double[,] weightsGradients;
        private double[] biasesGradients;
#endif

#endif
        #endregion


        #region Properties
        public override double DropoutParameter
        {
            set { this.dropoutParameter = value; }
        }

#if GRADIENT_CHECK
        // Accessors for gradient check

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
            this.outputNeurons = new Neurons(this.nOutputUnits);
        }

        public override void InitializeParameters()
        {
            // 1. Make sure this method is only call AFTER "SetupOutput()"
            base.InitializeParameters(); 

            // 2. Sample initial values:
            //  - Weigths are initialized as normally distributed numbers with mean 0 and std equals to 2/sqrt(nInputUnits)
            //  - Biases are initialized as normally distributed numbers with mean 0 and std 1

#if OPENCL_ENABLED
            float[,] initWeights = new float[outputNeurons.NumberOfUnits, inputNeurons.NumberOfUnits];
            float[] initBiases = new float[outputNeurons.NumberOfUnits];
#else
            this.weights = new double[outputNeurons.NumberOfUnits, inputNeurons.NumberOfUnits];
            this.biases = new double[outputNeurons.NumberOfUnits];
#endif

#if GRADIENT_CHECK
            this.weightsGradients = new double[this.OutputNeurons.NumberOfUnits, this.InputNeurons.NumberOfUnits];
            this.biasesGradients = new double[this.OutputNeurons.NumberOfUnits];
#endif

            double weightsStdDev = Math.Sqrt(2.0 / this.inputNeurons.NumberOfUnits);
            double uniformRand1;
            double uniformRand2;
            double tmp;

            for (int iRow = 0; iRow < outputNeurons.NumberOfUnits; iRow++)
            {

                for (int iCol = 0; iCol < inputNeurons.NumberOfUnits; iCol++)
                {
                    uniformRand1 = Global.rng.NextDouble();
                    uniformRand2 = Global.rng.NextDouble();
                    // Use a Box-Muller transform to get a random normal(0,1)
                    tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);
                    tmp = weightsStdDev * tmp; // rescale
#if OPENCL_ENABLED
                    initWeights[iRow, iCol] = (float)tmp;
#else
                    weights[iRow, iCol] = tmp;
#endif
                }
#if OPENCL_ENABLED
                initBiases[iRow] = 0.0001f;
#else
                biases[iRow] = 0.01;
#endif

            }

            // 3. If using OpenCL, create buffers for parameters and copy sampled initial values to them

#if OPENCL_ENABLED

            int weightBufferSize = sizeof(float) * (outputNeurons.NumberOfUnits * inputNeurons.NumberOfUnits);
            int biasesBufferSize = sizeof(float) * outputNeurons.NumberOfUnits;

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

            this.biasesSpeedGPU = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)biasesBufferSize,
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            // ...but better make extra sure and enforce this.
            OpenCLSpace.WipeBuffer(weightsSpeedGPU, (nInputUnits * nOutputUnits), typeof(float));
            OpenCLSpace.WipeBuffer(biasesSpeedGPU, nOutputUnits, typeof(float));


#else
            // Initialize updates speeds to zero

            this.weightsUpdateSpeed = new double[this.OutputNeurons.NumberOfUnits, this.InputNeurons.NumberOfUnits];
            this.biasesUpdateSpeed = new double[this.OutputNeurons.NumberOfUnits];

            // Also initialize dropout mask
            this.dropoutMask = new bool[nOutputUnits*inputNeurons.MiniBatchSize];
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

            int miniBatchSize = outputNeurons.MiniBatchSize;


            // FeedForward (2D) ________________________________________________________________________________
            
            // Local
            int optimalLocalWorkSize0 = OpenCLSpace.OPTIMAL_GROUP_SIZE / miniBatchSize;
            this.forwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)(optimalLocalWorkSize0), (IntPtr)miniBatchSize };

            // Global
            int smallestMultiple = (int)(optimalLocalWorkSize0 * Math.Ceiling((double)(nOutputUnits) / (double)optimalLocalWorkSize0));
            this.forwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple, (IntPtr)miniBatchSize };


            // BackPropagate (2D) _________________________________________________________________________________

            // Local
            this.backwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)(optimalLocalWorkSize0), (IntPtr)miniBatchSize };
            
            // Global
            smallestMultiple = (int)(optimalLocalWorkSize0 * Math.Ceiling((double)(nInputUnits) / (double)optimalLocalWorkSize0)); // input this time!
            this.backwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple, (IntPtr)miniBatchSize };


            // UpdateSpeeds and UpdateParameters (2D) ________________________________________________________________

            // Local
            optimalLocalWorkSize0 = OpenCLSpace.OPTIMAL_GROUP_SIZE / OpenCLSpace.BASE_GROUP_SIZE;
            this.updateLocalWorkSizePtr = new IntPtr[] { (IntPtr)optimalLocalWorkSize0, (IntPtr)OpenCLSpace.BASE_GROUP_SIZE }; // product is OPTIMAL_WORK_SIZE

            // Global
            int smallestMultiple0 = (int)(optimalLocalWorkSize0 * Math.Ceiling((double)(nOutputUnits) / (double)optimalLocalWorkSize0));
            int smallestMultiple1 = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(nInputUnits) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.updateGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple0, (IntPtr)smallestMultiple1 };
#endif
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
            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.FCForwardParallel, 0, outputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForwardParallel, 1, inputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForwardParallel, 2, weightsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForwardParallel, 3, biasesGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForwardParallel, 4, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForwardParallel, 5, (IntPtr)sizeof(int), nOutputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForwardParallel, 6, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForwardParallel, 7, (IntPtr)sizeof(float), (float)dropoutParameter);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForwardParallel, 8, (IntPtr)sizeof(ulong), (ulong)Guid.NewGuid().GetHashCode()); // this should be quite a good random seed
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.FeedForward(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.FCForwardParallel,
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
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackwardParallel, 0, inputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackwardParallel, 1, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackwardParallel, 2, weightsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackwardParallel, 3, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackwardParallel, 4, (IntPtr)sizeof(int), nOutputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackwardParallel, 5, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.BackPropagate(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.FCBackwardParallel,
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


        public override void UpdateSpeeds(double learningRate, double momentumCoefficient)
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
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeedsParallel, 0, weightsSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeedsParallel, 1, biasesSpeedGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeedsParallel, 2, inputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeedsParallel, 3, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeedsParallel, 4, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeedsParallel, 5, (IntPtr)sizeof(int), nOutputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeedsParallel, 6, (IntPtr)sizeof(float), (float)momentumCoefficient);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeedsParallel, 7, (IntPtr)sizeof(float), (float)learningRate);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeedsParallel, 8, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.UpdateSpeeds(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.FCUpdateSpeedsParallel,
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


#if DEBUGGING_STEPBYSTEP_FC
            /* ------------------------- DEBUGGING --------------------------------------------- */

            // Display weight update speed after update
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
            Console.WriteLine("\nWeight update speed AFTER update:");
            for (int i = 0; i < tmpWeightsUpdateSpeed.GetLength(0); i++)
            {
                for (int j = 0; j < tmpWeightsUpdateSpeed.GetLength(1); j++)
                    Console.Write("{0}  ", tmpWeightsUpdateSpeed[i, j]);
                Console.WriteLine();
            }
            Console.WriteLine();
            Console.ReadKey();
            

            // Display weights after update
            float[,] weightsAfterUpdate = new float[output.NumberOfUnits, input.NumberOfUnits];
#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                            weightsGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * input.NumberOfUnits * sizeof(float)),
                                            weightsAfterUpdate,  // destination
                                            0,
                                            null,
                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer weightsAfterUpdate");
#else
            weightsAfterUpdate = weights;
#endif
            Console.WriteLine("\nWeights AFTER update:");
            for (int i = 0; i < weightsAfterUpdate.GetLength(0); i++)
            {
                for (int j = 0; j < weightsAfterUpdate.GetLength(1); j++)
                    Console.Write("{0}  ", weightsAfterUpdate[i, j]);
                Console.WriteLine();
            }
            Console.WriteLine();
            Console.ReadKey();
            

            // Display biases after update
            float[] biasesAfterUpdate = new float[output.NumberOfUnits];
#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                            biasesGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * sizeof(float)),
                                            biasesAfterUpdate,  // destination
                                            0,
                                            null,
                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer biasesAfterUpdate");
#else
            biasesAfterUpdate = biases;
#endif
            Console.WriteLine("\nBiases AFTER update:");
            for (int i = 0; i < biasesAfterUpdate.Length; i++)
            {
                Console.Write("{0}  ", biasesAfterUpdate[i]);
            }
            Console.WriteLine();
            Console.ReadKey();


            /*------------------------- END DEBUGGING ---------------------------------------- */
#endif

#if TIMING_LAYERS
            Utils.FCUpdateSpeedsTimer.Stop();
#endif

        }

        public override void UpdateParameters(double weightDecayCoeff)
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
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 6, (IntPtr)sizeof(float), (float)weightDecayCoeff);
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

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
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

    }
}
