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

        // Host
        private float[,] weights;
        private float[] biases;

        private float[,] weightsUpdateSpeed;
        private float[] biasesUpdateSpeed;

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
#endif

        #endregion


        #region Properties

        public int NumberOfUnits // alias for NOutputUnits (property of parent class)
        {
            get { return nOutputUnits; }
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

        /// <summary>
        /// Connect layer to the previous one.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public override void ConnectTo(Layer PreviousLayer)
        {
            base.ConnectTo(PreviousLayer);
            this.outputNeurons = new Neurons(this.nOutputUnits);

#if OPENCL_ENABLED
            SetWorkGroupSizes();
#endif
        }

#if OPENCL_ENABLED
        private void SetWorkGroupSizes()
        {

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
        }
#endif
        
        public override void InitializeParameters() // only call after either "SetAsFirstLayer()" or "ConnectTo()"
        {
            base.InitializeParameters(); // ensures this method is only call AFTER "ConnectTo()"

            // Weigths are initialized as normally distributed numbers with mean 0 and std equals to 2/sqrt(nInputUnits)
            // Biases are initialized as normally distributed numbers with mean 0 and std 1

            // Host

            this.weights = new float[this.OutputNeurons.NumberOfUnits, this.InputNeurons.NumberOfUnits];
            this.biases = new float[this.OutputNeurons.NumberOfUnits];

            double weightsStdDev = Math.Sqrt(2.0/this.inputNeurons.NumberOfUnits);
            double uniformRand1;
            double uniformRand2;
            double tmp;

            for (int iRow = 0; iRow < this.weights.GetLength(0); iRow++)
            {
                
                for (int iCol = 0; iCol < this.weights.GetLength(1); iCol++)
                {
                    uniformRand1 = Global.rng.NextDouble();
                    uniformRand2 = Global.rng.NextDouble();
                    // Use a Box-Muller transform to get a random normal(0,1)
                    tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);
                    tmp = weightsStdDev * tmp; // rescale

                    weights[iRow, iCol] = (float)tmp;
                }

                biases[iRow] = 0.01F; // (float)tmp;
                
            }

            // Also initialize updates speeds to zero (for momentum)
            this.weightsUpdateSpeed = new float[this.OutputNeurons.NumberOfUnits, this.InputNeurons.NumberOfUnits];
            this.biasesUpdateSpeed = new float[this.OutputNeurons.NumberOfUnits];


#if OPENCL_ENABLED
            
            int weightBufferSize = sizeof(float) * (this.OutputNeurons.NumberOfUnits * this.InputNeurons.NumberOfUnits);
            int biasesBufferSize = sizeof(float) * this.OutputNeurons.NumberOfUnits;

            this.weightsGPU = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                    (IntPtr)weightBufferSize,
                                                    weights,
                                                    out OpenCLSpace.ClError);
            this.biasesGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                    (IntPtr)biasesBufferSize,
                                                    biases,
                                                    out OpenCLSpace.ClError);

            this.weightsSpeedGPU = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                                (IntPtr)weightBufferSize,
                                                                weightsUpdateSpeed,
                                                                out OpenCLSpace.ClError);
            this.biasesSpeedGPU = (Mem)Cl.CreateBuffer(   OpenCLSpace.Context,
                                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                                (IntPtr)biasesBufferSize,
                                                                biasesUpdateSpeed,
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InitializeParameters(): Cl.CreateBuffer");

            
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
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.FCForward, 0, outputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 1, inputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 2, weightsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 3, biasesGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 4, (IntPtr)sizeof(int), inputNeurons.NumberOfUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCForward, 5, (IntPtr)sizeof(int), outputNeurons.NumberOfUnits);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.FeedForward(): Cl.SetKernelArg");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
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

                float[] unbiasedOutput = Utils.MultiplyMatrixByVector(this.weights, this.inputNeurons.GetHost()[m]);
                this.outputNeurons.SetHost(m, unbiasedOutput.Zip(this.biases, (x, y) => x + y).ToArray());
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
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 0, InputNeurons.DeltaGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCBackward, 1, OutputNeurons.DeltaGPU[m]);
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
                this.inputNeurons.DeltaHost[m] = Utils.MultiplyMatrixTranspByVector(this.weights, this.outputNeurons.DeltaHost[m]);
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

            /* ------------------------- END DEBUGGING ---------------------------------------- */
#endif

#if DEBUGGING_STEPBYSTEP_FC
            /* ------------------------- DEBUGGING --------------------------------------------- */

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
            

            /*------------------------- END DEBUGGING ---------------------------------------- */
#endif

#if DEBUGGING_STEPBYSTEP_FC
            /* ------------------------- DEBUGGING --------------------------------------------- */

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
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 2, InputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateSpeeds, 3, OutputNeurons.DeltaGPU[m]);
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
                            this.weightsUpdateSpeed[i, j] *= (float)momentumCoefficient;

                        this.weightsUpdateSpeed[i, j] -= (float) learningRate/miniBatchSize * this.inputNeurons.GetHost()[m][j] * this.outputNeurons.DeltaHost[m][i];
                    }

                    // update biases
                    if (m == 0)
                            this.biasesUpdateSpeed[i] *= (float)momentumCoefficient;

                    this.biasesUpdateSpeed[i] -= (float) learningRate/miniBatchSize * this.outputNeurons.DeltaHost[m][i];
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
            
            /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif

#if DEBUGGING_STEPBYSTEP_FC
            /* ------------------------- DEBUGGING --------------------------------------------- */

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
            
            /* ------------------------- END DEBUGGING --------------------------------------------- */
#endif

#if DEBUGGING_STEPBYSTEP_FC
            /* ------------------------- DEBUGGING --------------------------------------------- */

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
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 0, weightsGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 1, biasesGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 2, weightsSpeedGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 3, biasesSpeedGPU);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 4, (IntPtr)sizeof(int), nInputUnits);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.FCUpdateParameters, 5, (IntPtr)sizeof(int), nOutputUnits);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "FullyConnected.UpdateParameters(): Cl.SetKernelArg");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
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
                        this.weights[i, j] += this.weightsUpdateSpeed[i, j];
                    }

                    // update biases
                    this.biases[i] += this.biasesUpdateSpeed[i];
                }
#endif
        }
        
        #endregion

    }
}
