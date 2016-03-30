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

        private Mem weightsGPU;
        private Mem biasesGPU;

        private Mem weightsUpdateSpeedGPU;
        private Mem biasesUpdateSpeedGPU;

        // Global and local work-group sizes (for OpenCL kernels) - will be set in SetWorkGroupSizes();
        private IntPtr[] forwardGlobalWorkSizePtr; //  = new IntPtr[] { (IntPtr)(Output.NumberOfUnits) };
        private IntPtr[] forwardLocalWorkSizePtr; // = new IntPtr[] { (IntPtr)Math.Min(128, Output.NumberOfUnits) }; // may need some fine-tuning
        private IntPtr[] backwardGlobalWorkSizePtr;
        private IntPtr[] backwardLocalWorkSizePtr;
        private IntPtr[] updateGlobalWorkSizePtr; // = new IntPtr[] { (IntPtr)(Output.NumberOfUnits), (IntPtr)(Input.NumberOfUnits) };
        private IntPtr[] updateLocalWorkSizePtr; // = new IntPtr[] { (IntPtr)Math.Min(128, Output.NumberOfUnits), (IntPtr)Math.Min(128, Input.NumberOfUnits) }; // may need some fine-tuning


#else


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
            this.numberOfUnits = nUnits;
            this.type = "FullyConnected";

           
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

            Random rng = new Random(); //reuse this if you are generating many
            double weightsStdDev = Math.Sqrt(2.0/this.input.NumberOfUnits);
            double uniformRand1;
            double uniformRand2;
            double tmp;

            for (int iRow = 0; iRow < this.weights.GetLength(0); iRow++)
            {
                
                for (int iCol = 0; iCol < this.weights.GetLength(1); iCol++)
                {
                    uniformRand1 = rng.NextDouble();
                    uniformRand2 = rng.NextDouble();
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

            this.weightsGPU = (Mem)Cl.CreateBuffer(CL.Context, MemFlags.ReadWrite, (IntPtr)weightBufferSize, out CL.Error);
            this.biasesGPU = (Mem)Cl.CreateBuffer(CL.Context, MemFlags.ReadWrite, (IntPtr)biasesBufferSize, out CL.Error);
            this.weightsUpdateSpeedGPU = (Mem)Cl.CreateBuffer(CL.Context, MemFlags.ReadWrite, (IntPtr)weightBufferSize, out CL.Error);
            this.biasesUpdateSpeedGPU = (Mem)Cl.CreateBuffer(CL.Context, MemFlags.ReadWrite, (IntPtr)biasesBufferSize, out CL.Error);
            CL.CheckErr(CL.Error, "InitializeParameters(): Cl.CreateBuffer");
#endif
        }


        private void SetWorkGroupSizes()
        {
            
            // FeedForward
            this.forwardGlobalWorkSizePtr = new IntPtr[] { (IntPtr)(Output.NumberOfUnits) };
            

            // BackPropagate


            // UpdateParameters
        }


        #endregion


        #region Training methods

        public override void FeedForward()
        {

#if OPENCL_ENABLED

            // Set kernel arguments
            CL.Error = Cl.SetKernelArg(CL.FCForward, 0, Output.ActivationsGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCForward, 1, Input.ActivationsGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCForward, 2, weightsGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCForward, 3, biasesGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCForward, 4, (IntPtr)sizeof(int), Input.NumberOfUnits);
            CL.Error |= Cl.SetKernelArg(CL.FCForward, 5, (IntPtr)sizeof(int), Output.NumberOfUnits);
            CL.CheckErr(CL.Error, "FullyConnected.FeedForward(): Cl.SetKernelArg");

            // Run kernel
            IntPtr[] globalWorkSizePtr = new IntPtr[] { (IntPtr)(Output.NumberOfUnits) };
            IntPtr[] localWorkSizePtr = new IntPtr[] { (IntPtr)Math.Min(128, Output.NumberOfUnits) }; // may need some fine-tuning
            CL.Error = Cl.EnqueueNDRangeKernel(CL.Queue, CL.FCForward, 1, null, globalWorkSizePtr, localWorkSizePtr, 0, null, out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnected.FeedForward(): Cl.EnqueueNDRangeKernel");

#else

            float[] unbiasedOutput = Utils.MultiplyMatrixByVector(this.weights, this.Input.GetHost());
            this.output.SetHost(unbiasedOutput.Zip(this.biases, (x, y) => x + y).ToArray());

#endif
        }


        public override void BackPropagate()
        {

#if OPENCL_ENABLED

            // Set kernel arguments
            CL.Error |= Cl.SetKernelArg(CL.FCBackward, 0, Input.DeltaGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCBackward, 1, Output.DeltaGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCBackward, 2, weightsGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCBackward, 3, (IntPtr)sizeof(int), Input.NumberOfUnits);
            CL.Error |= Cl.SetKernelArg(CL.FCBackward, 4, (IntPtr)sizeof(int), Output.NumberOfUnits);
            CL.CheckErr(CL.Error, "FullyConnected.BackPropagate(): Cl.SetKernelArg");

            // Run kernel
            IntPtr[] globalWorkSizePtr = new IntPtr[] { (IntPtr)(Input.NumberOfUnits) };
            IntPtr[] localWorkSizePtr = new IntPtr[] { (IntPtr)Math.Min(128, Input.NumberOfUnits) }; // may need some fine-tuning
            CL.Error = Cl.EnqueueNDRangeKernel(CL.Queue, CL.FCBackward, 1, null, globalWorkSizePtr, localWorkSizePtr, 0, null, out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnected.BackPropagate(): Cl.EnqueueNDRangeKernel");

#else
            this.Input.DeltaHost = Utils.MultiplyMatrixTranspByVector(this.weights, this.Output.DeltaHost);
#endif
        }

        public override void UpdateParameters(double learningRate, double momentumCoefficient)
        {

#if OPENCL_ENABLED

            // Set kernel arguments
            CL.Error |= Cl.SetKernelArg(CL.FCUpdateParameters, 0, weightsGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCUpdateParameters, 1, biasesGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCUpdateParameters, 2, weightsUpdateSpeedGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCUpdateParameters, 3, biasesUpdateSpeedGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCUpdateParameters, 4, Input.ActivationsGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCUpdateParameters, 5, Output.DeltaGPU);
            CL.Error |= Cl.SetKernelArg(CL.FCUpdateParameters, 6, (IntPtr)sizeof(int), Input.NumberOfUnits);
            CL.Error |= Cl.SetKernelArg(CL.FCUpdateParameters, 7, (IntPtr)sizeof(int), Output.NumberOfUnits);
            CL.Error |= Cl.SetKernelArg(CL.FCUpdateParameters, 8, (IntPtr)sizeof(float), (float) learningRate);
            CL.Error |= Cl.SetKernelArg(CL.FCUpdateParameters, 9, (IntPtr)sizeof(float), (float) momentumCoefficient);
            CL.CheckErr(CL.Error, "FullyConnected.UpdateParameters(): Cl.SetKernelArg");

            // Run kernel
            IntPtr[] globalWorkSizePtr = new IntPtr[] { (IntPtr)(Output.NumberOfUnits), (IntPtr)(Input.NumberOfUnits) };
            IntPtr[] localWorkSizePtr = new IntPtr[] { (IntPtr)Math.Min(128, Output.NumberOfUnits), (IntPtr)Math.Min(128, Input.NumberOfUnits) }; // may need some fine-tuning
            CL.Error = Cl.EnqueueNDRangeKernel(CL.Queue, CL.FCUpdateParameters, 2, null, globalWorkSizePtr, localWorkSizePtr, 0, null, out CL.Event);
            CL.CheckErr(CL.Error, "FullyConnected.UpdateParameters(): Cl.EnqueueNDRangeKernel");

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
        }

        #endregion


        #region Debugging/helper methods

        public override void DisplayParameters()
        {
            Console.WriteLine("\n\n ======== LAYER =========\n\n");

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

        }
        #endregion


    }
}
