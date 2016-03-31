using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    class SoftMax : Layer
    {
        #region SoftMax layer class fields (private)


#if OPENCL_ENABLED

        Mem auxiliaryFloatBuffer; // needed by forward pass

        private IntPtr[] globalWorkSizePtr;
        private IntPtr[] localWorkSizePtr;
        // in this case nInput = nOutput  ==>  only need to set one global/local work size 
        // (i.e. no need to distinguish between forward and backward pass)
#endif

        #endregion


        #region Setup methods (to be called once)

        /// <summary>
        /// Constructor of Softmax layer.
        /// </summary>
        /// <param name="Beta"></param>
        public SoftMax()
        {
            //Console.WriteLine("Adding a SoftMax layer...");

            this.type = "SoftMax";
        }

        /// <summary>
        ///  Connect current layer to layer given as argument.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public override void ConnectTo(Layer PreviousLayer)
        {
            base.ConnectTo(PreviousLayer);

            this.numberOfUnits = PreviousLayer.Output.NumberOfUnits;
            this.output = new Neurons(this.numberOfUnits);

        }

        /// <summary>
        /// Method to set this layer as the first layer of the network.
        /// </summary>
        public override void SetAsFirstLayer(int InputWidth, int InputHeight, int InputDepth)
        {
            throw new System.InvalidOperationException("You are setting a SoftMax layer as first layer of the network...\nIs it really what you want to do?");
        }

        public override void InitializeParameters()
        {
#if OPENCL_ENABLED
            this.auxiliaryFloatBuffer = (Mem)Cl.CreateBuffer(CL.Context, MemFlags.ReadWrite, (IntPtr)sizeof(float), out CL.Error);
            CL.CheckErr(CL.Error, "Cl.CreateBuffer auxiliaryFloatBuffer");

            SetWorkGroupSizes();
#endif
        }

        private void SetWorkGroupSizes()
        {
            // Work group sizes will be set as follows:
            //      global work size = total number of processes needed
            //      local work size = largest divisor of global work size <= maxWorkGroupSize of device in context
            // (this is probably suboptimal, but improvements are most likely negligible compared to improvements elsewhere, e.g. in the kernels code)

            this.globalWorkSizePtr = new IntPtr[] { (IntPtr)(Output.NumberOfUnits) };
            int tmpLocalWorkSize = Output.NumberOfUnits;
            while (tmpLocalWorkSize > CL.maxWorkGroupSize || tmpLocalWorkSize > CL.maxWorkItemSizes[0])
                tmpLocalWorkSize /= 2;
            this.localWorkSizePtr = new IntPtr[] { (IntPtr)(tmpLocalWorkSize) };
        }

        #endregion


        #region Training methods

        public override void FeedForward()
        {
#if OPENCL_ENABLED

            // Set kernel arguments
            CL.Error  = Cl.SetKernelArg(CL.SoftmaxForward, 0, Output.ActivationsGPU);
            CL.Error |= Cl.SetKernelArg(CL.SoftmaxForward, 1, Input.ActivationsGPU);
            CL.Error |= Cl.SetKernelArg(CL.SoftmaxForward, 2, auxiliaryFloatBuffer);
            CL.Error |= Cl.SetKernelArg(CL.SoftmaxForward, 3, (IntPtr)sizeof(int), Output.NumberOfUnits);
            CL.CheckErr(CL.Error, "Softmax.FeedForward(): Cl.SetKernelArg");

            // Run kernel
            CL.Error = Cl.EnqueueNDRangeKernel( CL.Queue,
                                                CL.SoftmaxForward,
                                                1,
                                                null,
                                                globalWorkSizePtr,
                                                localWorkSizePtr,
                                                0,
                                                null,
                                                out CL.Event);
            CL.CheckErr(CL.Error, "Softmax.FeedForward(): Cl.EnqueueNDRangeKernel");

#else

            // use rescaling trick to improve numerical stability
            float maxInput = this.input.GetHost()[0];
            for (int i = 1; i < this.numberOfUnits; i++)
            {
                if (this.input.GetHost()[i] > maxInput)
                    maxInput = this.input.GetHost()[i];
            }

            float[] tmpOutput = new float[this.numberOfUnits];
            for (int i = 0; i < this.numberOfUnits; i++)
            {
                tmpOutput[i] = (float)Math.Exp(this.input.GetHost()[i]-maxInput);
            }
            float sum = tmpOutput.Sum();
            for (int i = 0; i < this.numberOfUnits; i++)
            {
                tmpOutput[i] /= sum;
            }

            this.output.SetHost(tmpOutput);
#endif
        }


        public override void BackPropagate()
        {
            if (this.Input.DeltaHost.Length != this.Output.DeltaHost.Length)
                throw new System.InvalidOperationException("Softmax layer: mismatch in length of delta arrays.");

            // NO backprop here!!
            // Compute directly input.Delta from cross-entropy cost: faster and numerically more stable
        }

        #endregion


    }
}
