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

        #region Fields (private)

        // Host
        private float[,] weights;
        private float[] biases;

        private float[,] weightsUpdateSpeed;
        private float[] biasesUpdateSpeed;

#if OPENCL_ENABLED

        private Kernel ForwardKernel;
        private Kernel BackwardKernel;
        private Kernel UpdateKernel;

        private Mem weightsGPU;
        private Mem biasesGPU;

        private Mem weightsUpdateSpeedGPU;
        private Mem biasesUpdateSpeedGPU;

        // Global and local work-group sizes (for OpenCL kernels) - will be set in SetWorkGroupSizes();
        private IntPtr[] forwardGlobalWorkSizePtr;
        private IntPtr[] forwardLocalWorkSizePtr;
        private IntPtr[] backwardGlobalWorkSizePtr;
        private IntPtr[] backwardLocalWorkSizePtr;
        private IntPtr[] updateGlobalWorkSizePtr;
        private IntPtr[] updateLocalWorkSizePtr;
#endif

        #endregion


        #region Properties (public)

        public int NumberOfUnits
        {
            get { return numberOfUnits; }
        }

        #endregion


        #region Setup methods (to be called once)

        /// <summary>
        /// Constructor of fully connected layer type. Specify number of units as argument.
        /// </summary>
        /// <param name="nUnits"></param>
        public FullyConnectedLayer(int nUnits)
        {
            this.type = "FullyConnected";
            this.numberOfUnits = nUnits;

#if OPENCL_ENABLED
            // Load and build kernels
            ForwardKernel = CL.LoadBuildKernel(CL.KernelsPath + "/FCForward.cl", "FCForward");
            BackwardKernel = CL.LoadBuildKernel(CL.KernelsPath + "/FCBackward.cl", "FCBackward");
            UpdateKernel = CL.LoadBuildKernel(CL.KernelsPath + "/FCUpdateParameters.cl", "FCUpdateParameters");
#endif
        }

        /// <summary>
        /// Set this layer as the first layer of the neural network.
        /// </summary>
        /// <param name="InputWidth"></param>
        /// <param name="InputHeight"></param>
        /// <param name="InputDepth"></param>
        public override void SetAsFirstLayer(int InputWidth, int InputHeight, int InputDepth)
        {
            this.input = new Neurons(InputWidth * InputHeight * InputDepth);
            this.output = new Neurons(this.numberOfUnits);
        }

        /// <summary>
        /// Connect layer to the previous one.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public override void ConnectTo(Layer PreviousLayer)
        {
            base.ConnectTo(PreviousLayer);
            this.output = new Neurons(this.numberOfUnits);
        }

        
        public override void InitializeParameters() // only call after either "SetAsFirstLayer()" or "ConnectTo()"
        {
            // Weigths are initialized as normally distributed numbers with mean 0 and std equals to 2/sqrt(nInputUnits)
            // Biases are initialized as normally distributed numbers with mean 0 and std 1

            // Host

            this.weights = new float[this.Output.NumberOfUnits, this.Input.NumberOfUnits];
            this.biases = new float[this.Output.NumberOfUnits];

            double weightsStdDev = Math.Sqrt(2.0/this.input.NumberOfUnits);
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

                //uniformRand1 = rng.NextDouble();
                //uniformRand2 = rng.NextDouble();
                // Use a Box-Muller transform to get a random normal(0,1)
                //tmp = Math.Sqrt(-2.0 * Math.Log(uniformRand1)) * Math.Sin(2.0 * Math.PI * uniformRand2);

                biases[iRow] = 0.01F; // (float)tmp;
                
            }

            // Also initialize updates speeds to zero (for momentum)
            this.weightsUpdateSpeed = new float[this.Output.NumberOfUnits, this.Input.NumberOfUnits];
            this.biasesUpdateSpeed = new float[this.Output.NumberOfUnits];


#if OPENCL_ENABLED
            
            int weightBufferSize = sizeof(float) * (this.Output.NumberOfUnits * this.Input.NumberOfUnits);
            int biasesBufferSize = sizeof(float) * this.Output.NumberOfUnits;

            this.weightsGPU = (Mem)Cl.CreateBuffer( CL.Context, 
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr, 
                                                    (IntPtr)weightBufferSize, 
                                                    weights, 
                                                    out CL.Error);
            this.biasesGPU = (Mem)Cl.CreateBuffer(  CL.Context, 
                                                    MemFlags.ReadWrite | MemFlags.CopyHostPtr, 
                                                    (IntPtr)biasesBufferSize, 
                                                    biases, 
                                                    out CL.Error);

            this.weightsUpdateSpeedGPU = (Mem)Cl.CreateBuffer(  CL.Context, 
                                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr, 
                                                                (IntPtr)weightBufferSize, 
                                                                weightsUpdateSpeed,
                                                                out CL.Error);
            this.biasesUpdateSpeedGPU = (Mem)Cl.CreateBuffer(   CL.Context, 
                                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr, 
                                                                (IntPtr)biasesBufferSize, 
                                                                biasesUpdateSpeed,
                                                                out CL.Error);
            CL.CheckErr(CL.Error, "InitializeParameters(): Cl.CreateBuffer");

            SetWorkGroupSizes();
#endif
        }

#if OPENCL_ENABLED
        private void SetWorkGroupSizes()
        {
            // Work group sizes will be set as follows:
            //      global work size = total number of processes needed
            //      local work size = largest divisor of global work size <= maxWorkGroupSize of device in context
            // (this is probably suboptimal, but improvements are most likely negligible compared to improvements elsewhere, e.g. in the kernels code)

            // FeedForward
            this.forwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(Output.NumberOfUnits) }; 
            int tmpFwLocalWorkSize = Output.NumberOfUnits; // 
            while (tmpFwLocalWorkSize > CL.MaxWorkGroupSize || tmpFwLocalWorkSize > CL.MaxWorkItemSizes[0]) 
                tmpFwLocalWorkSize /= 2;
            this.forwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)(tmpFwLocalWorkSize) };

            // BackPropagate
            this.backwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(Input.NumberOfUnits) };
            int tmpBwLocalWorkSize = Input.NumberOfUnits;
            while (tmpBwLocalWorkSize > CL.MaxWorkGroupSize || tmpBwLocalWorkSize > CL.MaxWorkItemSizes[0]) 
                tmpBwLocalWorkSize /= 2;
            this.backwardLocalWorkSizePtr = new IntPtr[] { (IntPtr)(tmpBwLocalWorkSize) };
           
            // UpdateParameters
            this.updateGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(Output.NumberOfUnits), (IntPtr)(Input.NumberOfUnits) };
            int[] tmpUpdLocalWorkSize = new int[] { Output.NumberOfUnits, Input.NumberOfUnits };
            // make each local work group dimension <= corresponding max work item size (depends on device)
            while (tmpUpdLocalWorkSize[0] > CL.MaxWorkItemSizes[0] && tmpUpdLocalWorkSize[0] % 2 == 0)
                    tmpUpdLocalWorkSize[0] /= 2;
            while (tmpUpdLocalWorkSize[1] > CL.MaxWorkItemSizes[1] && tmpUpdLocalWorkSize[1] % 2 == 0)
                tmpUpdLocalWorkSize[1] /= 2;
            // make entire local work group size (i.e. product of dimensions) <= of max work group size (depends on device)
            while (tmpUpdLocalWorkSize[0] * tmpUpdLocalWorkSize[1] > CL.MaxWorkGroupSize && tmpUpdLocalWorkSize[1] % 2 == 0)
            {
                tmpUpdLocalWorkSize[1] /= 2;
            }
            while (tmpUpdLocalWorkSize[0] * tmpUpdLocalWorkSize[1] > CL.MaxWorkGroupSize && tmpUpdLocalWorkSize[0] % 2 == 0)
            {
                tmpUpdLocalWorkSize[0] /= 2;
                if (tmpUpdLocalWorkSize[0] == 1)
                {
                    throw new System.InvalidOperationException("I can't set a suitable local work group size! :(");
                }
            }
            this.updateLocalWorkSizePtr = new IntPtr[] { (IntPtr)(tmpUpdLocalWorkSize[0]), (IntPtr)(tmpUpdLocalWorkSize[1]) };

        }
#endif

        #endregion


        #region Training methods

        public override void FeedForward()
        {

#if OPENCL_ENABLED
            // Set kernel arguments
            CL.Error = Cl.SetKernelArg(ForwardKernel, 0, Output.ActivationsGPU);
            CL.Error |= Cl.SetKernelArg(ForwardKernel, 1, Input.ActivationsGPU);
            CL.Error |= Cl.SetKernelArg(ForwardKernel, 2, weightsGPU);
            CL.Error |= Cl.SetKernelArg(ForwardKernel, 3, biasesGPU);
            CL.Error |= Cl.SetKernelArg(ForwardKernel, 4, (IntPtr)sizeof(int), Input.NumberOfUnits);
            CL.Error |= Cl.SetKernelArg(ForwardKernel, 5, (IntPtr)sizeof(int), Output.NumberOfUnits);
            CL.CheckErr(CL.Error, "FullyConnected.FeedForward(): Cl.SetKernelArg");

            // Run kernel
            CL.Error = Cl.EnqueueNDRangeKernel( CL.Queue,
                                                ForwardKernel, 
                                                1, 
                                                null, 
                                                forwardGlobalWorkSizePtr, 
                                                forwardLocalWorkSizePtr, 
                                                0, 
                                                null,
                                                out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnected.FeedForward(): Cl.EnqueueNDRangeKernel");

            CL.Error = Cl.Finish(CL.Queue);
            CL.CheckErr(CL.Error, "Cl.Finish");

            CL.Error = Cl.ReleaseEvent(CL.Event);
            CL.CheckErr(CL.Error, "Cl.ReleaseEvent");
#else

            float[] unbiasedOutput = Utils.MultiplyMatrixByVector(this.weights, this.Input.GetHost());
            this.output.SetHost(unbiasedOutput.Zip(this.biases, (x, y) => x + y).ToArray());

#endif
        }

        public override void BackPropagate()
        {

#if OPENCL_ENABLED

            // Set kernel arguments
            CL.Error |= Cl.SetKernelArg(BackwardKernel, 0, Input.DeltaGPU);
            CL.Error |= Cl.SetKernelArg(BackwardKernel, 1, Output.DeltaGPU);
            CL.Error |= Cl.SetKernelArg(BackwardKernel, 2, weightsGPU);
            CL.Error |= Cl.SetKernelArg(BackwardKernel, 3, (IntPtr)sizeof(int), Input.NumberOfUnits);
            CL.Error |= Cl.SetKernelArg(BackwardKernel, 4, (IntPtr)sizeof(int), Output.NumberOfUnits);
            CL.CheckErr(CL.Error, "FullyConnected.BackPropagate(): Cl.SetKernelArg");

            // Run kernel
            CL.Error = Cl.EnqueueNDRangeKernel( CL.Queue,
                                                BackwardKernel, 
                                                1, 
                                                null, 
                                                backwardGlobalWorkSizePtr, 
                                                backwardLocalWorkSizePtr, 
                                                0, 
                                                null, 
                                                out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnected.BackPropagate(): Cl.EnqueueNDRangeKernel");

            CL.Error = Cl.Finish(CL.Queue);
            CL.CheckErr(CL.Error, "Cl.Finish");

            CL.Error = Cl.ReleaseEvent(CL.Event);
            CL.CheckErr(CL.Error, "Cl.ReleaseEvent");
#else
            this.Input.DeltaHost = Utils.MultiplyMatrixTranspByVector(this.weights, this.Output.DeltaHost);
#endif
        }


        public override void UpdateParameters(double learningRate, double momentumCoefficient)
        {

#if DEBUGGING_STEPBYSTEP
            float[,] weightsBeforeUpdate = new float[output.NumberOfUnits, input.NumberOfUnits];
            /* ------------------------- DEBUGGING --------------------------------------------- */
#if OPENCL_ENABLED
            // Display weights before update
            
            CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                            weightsGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * input.NumberOfUnits * sizeof(float)),
                                            weightsBeforeUpdate,  // destination
                                            0,
                                            null,
                                            out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer weightsBeforeUpdate");
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

#if DEBUGGING_STEPBYSTEP
            /* ------------------------- DEBUGGING --------------------------------------------- */

            // Display biases before update
            float[] biasesBeforeUpdate = new float[output.NumberOfUnits];
#if OPENCL_ENABLED
            CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                            biasesGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * sizeof(float)),
                                            biasesBeforeUpdate,  // destination
                                            0,
                                            null,
                                            out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer biasesBeforeUpdate");
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

#if DEBUGGING_STEPBYSTEP
            /* ------------------------- DEBUGGING --------------------------------------------- */

            // Display weight update speed before update
            
            float[,] tmpWeightsUpdateSpeed = new float[output.NumberOfUnits, input.NumberOfUnits];
#if OPENCL_ENABLED
            CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                            weightsUpdateSpeedGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * input.NumberOfUnits * sizeof(float)),
                                            tmpWeightsUpdateSpeed,  // destination
                                            0,
                                            null,
                                            out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer weightsUpdateSpeed");
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
            CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                            input.ActivationsGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(input.NumberOfUnits * sizeof(float)),
                                            inputActivations,  // destination
                                            0,
                                            null,
                                            out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer inputActivations");

            Console.WriteLine("\nInput activations BEFORE update:");

            for (int j = 0; j < inputActivations.Length; j++)
            {
                Console.Write("{0}  ", inputActivations[j]);
            }
            Console.WriteLine();
            Console.ReadKey();
            


            // Display output delta before update

            float[] outputDelta = new float[output.NumberOfUnits];
            CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                            output.DeltaGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * sizeof(float)),
                                            outputDelta,  // destination
                                            0,
                                            null,
                                            out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer outputDelta");

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
            CL.Error  = Cl.SetKernelArg(UpdateKernel, 0, weightsGPU);
            CL.Error |= Cl.SetKernelArg(UpdateKernel, 1, biasesGPU);
            CL.Error |= Cl.SetKernelArg(UpdateKernel, 2, weightsUpdateSpeedGPU);
            CL.Error |= Cl.SetKernelArg(UpdateKernel, 3, biasesUpdateSpeedGPU);
            CL.Error |= Cl.SetKernelArg(UpdateKernel, 4, Input.ActivationsGPU);
            CL.Error |= Cl.SetKernelArg(UpdateKernel, 5, Output.DeltaGPU);
            CL.Error |= Cl.SetKernelArg(UpdateKernel, 6, (IntPtr)sizeof(int), Input.NumberOfUnits);
            CL.Error |= Cl.SetKernelArg(UpdateKernel, 7, (IntPtr)sizeof(int), Output.NumberOfUnits);
            CL.Error |= Cl.SetKernelArg(UpdateKernel, 8, (IntPtr)sizeof(float), (float)learningRate);
            CL.Error |= Cl.SetKernelArg(UpdateKernel, 9, (IntPtr)sizeof(float), (float)momentumCoefficient);
            CL.CheckErr(CL.Error, "FullyConnected.UpdateParameters(): Cl.SetKernelArg");

            // Run kernel
            CL.Error = Cl.EnqueueNDRangeKernel( CL.Queue,
                                                UpdateKernel, 
                                                2, 
                                                null, 
                                                updateGlobalWorkSizePtr, 
                                                updateLocalWorkSizePtr, 
                                                0, 
                                                null, 
                                                out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnected.UpdateParameters(): Cl.EnqueueNDRangeKernel");

            CL.Error = Cl.Finish(CL.Queue);
            CL.CheckErr(CL.Error, "Cl.Finish");

            CL.Error = Cl.ReleaseEvent(CL.Event);
            CL.CheckErr(CL.Error, "Cl.ReleaseEvent");
#else
            // Update weights
            for (int i = 0; i < this.weights.GetLength(0); i++)
            {
                for (int j = 0; j < this.weights.GetLength(1); j++)
                {
                    this.weightsUpdateSpeed[i, j] *= (float)momentumCoefficient;
                    this.weightsUpdateSpeed[i, j] -= (float) learningRate * this.input.GetHost()[j] * this.output.DeltaHost[i];

                    this.weights[i, j] += this.weightsUpdateSpeed[i, j];
                }
            }

            // Update biases
            for (int i = 0; i < this.biases.GetLength(0); i++)
            {
                this.biasesUpdateSpeed[i] *= (float)momentumCoefficient;
                this.biasesUpdateSpeed[i] -= (float) learningRate * this.output.DeltaHost[i];

                this.biases[i] += this.biasesUpdateSpeed[i];
            }
#endif

#if DEBUGGING_STEPBYSTEP
            /* ------------------------- DEBUGGING --------------------------------------------- */

            // Display weight update speed after update
#if OPENCL_ENABLED
            CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                            weightsUpdateSpeedGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * input.NumberOfUnits * sizeof(float)),
                                            tmpWeightsUpdateSpeed,  // destination
                                            0,
                                            null,
                                            out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer weightsUpdateSpeed");
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

#if DEBUGGING_STEPBYSTEP
            /* ------------------------- DEBUGGING --------------------------------------------- */

            // Display weights after update
            float[,] weightsAfterUpdate = new float[output.NumberOfUnits, input.NumberOfUnits];
#if OPENCL_ENABLED
            CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                            weightsGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * input.NumberOfUnits * sizeof(float)),
                                            weightsAfterUpdate,  // destination
                                            0,
                                            null,
                                            out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer weightsAfterUpdate");
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

#if DEBUGGING_STEPBYSTEP
            /* ------------------------- DEBUGGING --------------------------------------------- */

            // Display biases after update
            float[] biasesAfterUpdate = new float[output.NumberOfUnits];
#if OPENCL_ENABLED
            CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                            biasesGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * sizeof(float)),
                                            biasesAfterUpdate,  // destination
                                            0,
                                            null,
                                            out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer biasesAfterUpdate");
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

        #endregion


        #region Debugging/helper methods

        public override void DisplayParameters()
        {
            Console.WriteLine("\n\n ======== LAYER =========\n\n");
#if OPENCL_ENABLED

            float[,] finalWeigths = new float[output.NumberOfUnits, input.NumberOfUnits];
            CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                            weightsGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * input.NumberOfUnits * sizeof(float)),
                                            finalWeigths,  // destination
                                            0,
                                            null,
                                            out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer finalWeigths");
            Console.WriteLine("\nFinal weights:");
            for (int i = 0; i < finalWeigths.GetLength(0); i++)
            {
                for (int j = 0; j < finalWeigths.GetLength(1); j++)
                    Console.Write("{0}  ", finalWeigths[i, j]);
                Console.WriteLine();
            }
            Console.WriteLine();


            float[] finalBiases = new float[output.NumberOfUnits];
            CL.Error = Cl.EnqueueReadBuffer(CL.Queue,
                                            biasesGPU, // source
                                            Bool.True,
                                            (IntPtr)0,
                                            (IntPtr)(output.NumberOfUnits * sizeof(float)),
                                            finalBiases,  // destination
                                            0,
                                            null,
                                            out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnectedLayer.UpdateParameters Cl.clEnqueueReadBuffer finalBiases");
            Console.WriteLine("\nFinal biases:");
            for (int i = 0; i < finalBiases.GetLength(0); i++)
            {
                Console.Write("{0}  ", finalBiases[i]);

                Console.WriteLine();
            }
            Console.WriteLine();


#else
            Console.WriteLine("\nFinal weights:");
            for (int i = 0; i < weights.GetLength(0); i++ )
            {
                for (int j = 0; j < weights.GetLength(1); j++)
                    Console.Write("{0}  ", weights[i,j]);

                Console.WriteLine();
            }

            Console.WriteLine("\nFinal biases:");
            for (int i = 0; i < biases.GetLength(0); i++)
            {
                Console.Write("{0}  ", biases[i]);

                Console.WriteLine();
            }
#endif

        }
        #endregion


    }
}
