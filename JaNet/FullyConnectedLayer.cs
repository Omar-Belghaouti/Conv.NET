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

        // only "layer type-specific" field is the number of output units, 
        // which is also a field of base class Layer

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

#if GRADIENT_CHECK
        private double[,] weightsGradients;
        private double[] biasesGradients;
#endif

#endif
        #endregion


        #region Properties

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
                initBiases[iRow] = 0.01f;
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
#endif
        }


        public override void SetWorkGroups()
        {
#if OPENCL_ENABLED
            // TODO: update this using OutputNeurons.MiniBatchSize

            // Work group sizes will be set as follows:
            //      global work size = smallest multiple of BASE_GROUP_SIZE larger than the total number of processes needed (for efficiency)
            //      local work size = BASE_GROUP_SIZE or small multiples of it (making sure that global worksize is a multiple of this)
            // BASE_GROUP_SIZE is a constant multiple of 2. Suggested values: 32 (Nvidia) or 64 (AMD).

            // FeedForward (1D) ________________________________________________________________________________
            
            // Global
            int totalWorkItemsNeeded = nOutputUnits;
            int smallestMultipleOfBGS = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(totalWorkItemsNeeded) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.forwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(smallestMultipleOfBGS) };
            
            // Local
            int maxFWDKernelWorkGroupSize = Cl.GetKernelWorkGroupInfo(  OpenCLSpace.FCForward,
                                                                        OpenCLSpace.Device,
                                                                        KernelWorkGroupInfo.WorkGroupSize,
                                                                        out OpenCLSpace.ClError).CastTo<int>();
            int localWorkSize = OpenCLSpace.BASE_GROUP_SIZE;
            while (true)
            {
                int tmpLocalWorkSize = 2 * localWorkSize;

                bool globalDividesLocal = smallestMultipleOfBGS % tmpLocalWorkSize == 0;
                bool isLocalGroupTooLarge = tmpLocalWorkSize > OpenCLSpace.MaxWorkGroupSize;
                isLocalGroupTooLarge |= tmpLocalWorkSize > maxFWDKernelWorkGroupSize;

                if (globalDividesLocal && !isLocalGroupTooLarge) // if global divides local and it's not too large
                    localWorkSize = tmpLocalWorkSize;
                else
                    break;
            }
            this.forwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)(localWorkSize) };


            // BackPropagate (1D) _________________________________________________________________________________
            
            // Global
            totalWorkItemsNeeded = nInputUnits;
            smallestMultipleOfBGS = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(totalWorkItemsNeeded) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.backwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultipleOfBGS };

            // Local
            int maxBWDKernelWorkGroupSize = Cl.GetKernelWorkGroupInfo(  OpenCLSpace.FCBackward,
                                                                        OpenCLSpace.Device,
                                                                        KernelWorkGroupInfo.WorkGroupSize,
                                                                        out OpenCLSpace.ClError).CastTo<int>();
            localWorkSize = OpenCLSpace.BASE_GROUP_SIZE;


            while (true)
            {
                int tmpLocalWorkSize = 2 * localWorkSize;

                bool globalDividesLocal = smallestMultipleOfBGS % tmpLocalWorkSize == 0;
                bool isLocalGroupTooLarge = tmpLocalWorkSize > OpenCLSpace.MaxWorkGroupSize;
                isLocalGroupTooLarge |= tmpLocalWorkSize > maxBWDKernelWorkGroupSize;

                if (globalDividesLocal && !isLocalGroupTooLarge) // if global divides local and it's not too large
                    localWorkSize = tmpLocalWorkSize;
                else
                    break;
            }
            this.backwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)(localWorkSize) };


            // UpdateSpeeds and UpdateParameters (2D) ________________________________________________________________

            // Global
            int[] updTotalWorkItemsNeeded = new int[] { nOutputUnits, nInputUnits};
            int[] updSmallestMultipleOfBGS = new int[] {
                (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(updTotalWorkItemsNeeded[0]) / (double)OpenCLSpace.BASE_GROUP_SIZE)),
                (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(updTotalWorkItemsNeeded[1]) / (double)OpenCLSpace.BASE_GROUP_SIZE))
            };
            this.updateGlobalWorkSizePtr = new IntPtr[] { (IntPtr)updSmallestMultipleOfBGS[0], (IntPtr)updSmallestMultipleOfBGS[1] };

            // Local
            this.updateLocalWorkSizePtr = new IntPtr[] { (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE / 4), (IntPtr)(OpenCLSpace.BASE_GROUP_SIZE / 2) };
#endif
        }

        #endregion


        #region Methods

        public override void FeedForward()
        {
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
#if OPENCL_ENABLED
                // Set kernel arguments
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.FCForward, 0, outputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 1, inputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 2, weightsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 3, biasesGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 4, (IntPtr)sizeof(int), inputNeurons.NumberOfUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 5, (IntPtr)sizeof(int), outputNeurons.NumberOfUnits);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.FeedForward(): Cl.SetKernelArg");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                                OpenCLSpace.FCForward,
                                                                1,
                                                                null,
                                                                forwardGlobalWorkSizePtr,
                                                                forwardLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.FeedForward(): Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
#else

                double[] unbiasedOutput = Utils.MultiplyMatrixByVector(weights, inputNeurons.GetHost()[m]);
                this.outputNeurons.SetHost(m, unbiasedOutput.Zip(biases, (x, y) => x + y).ToArray());
#endif
            }

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

                // Set kernel arguments
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 0, InputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 1, OutputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 2, weightsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 3, (IntPtr)sizeof(int), InputNeurons.NumberOfUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 4, (IntPtr)sizeof(int), OutputNeurons.NumberOfUnits);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.BackPropagate(): Cl.SetKernelArg");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                                OpenCLSpace.FCBackward,
                                                                1,
                                                                null,
                                                                backwardGlobalWorkSizePtr,
                                                                backwardLocalWorkSizePtr,
                                                                0,
                                                                null,
                                                                out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.BackPropagate(): Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
#else
                inputNeurons.DeltaHost[m] = Utils.MultiplyMatrixTranspByVector(weights, outputNeurons.DeltaHost[m]);
#endif
            }

#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#endif
        }


        public override void UpdateSpeeds(double learningRate, double momentumCoefficient)
        {
            int miniBatchSize = inputNeurons.MiniBatchSize;

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
            for (int m = 0; m < miniBatchSize; m++)
            {
#if OPENCL_ENABLED
                // Set kernel arguments
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 0, weightsSpeedGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 1, biasesSpeedGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 2, InputNeurons.ActivationsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 3, OutputNeurons.DeltaGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 4, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 5, (IntPtr)sizeof(int), nOutputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 6, (IntPtr)sizeof(float), (float)momentumCoefficient);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 7, (IntPtr)sizeof(float), (float)(learningRate / miniBatchSize));
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 8, (IntPtr)sizeof(int), m);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.UpdateSpeeds(): Cl.SetKernelArg");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
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
#else


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
            } // end loop over mini-batch

#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#endif

        } 

        public override void UpdateParameters()
        {
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
        }
        
        #endregion

    }
}
