using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    class ReLU : Layer
    {
        #region ReLU layer class fields (private)

#if OPENCL_ENABLED

        private Kernel ForwardKernel;
        private Kernel BackwardKernel;

        private IntPtr[] globalWorkSizePtr;
        private IntPtr[] localWorkSizePtr; 
        // in this case nInput = nOutput  ==>  only need to set one global/local work size 
        // (i.e. no need to distinguish between forward and backward pass)
#endif

        #endregion


        #region Setup methods (to be called once)

        /// <summary>
        /// Constructor of ReLU layer.
        /// </summary>
        public ReLU()
        {
            //Console.WriteLine("Adding a ReLU layer...");
            this.type = "ReLU";

#if OPENCL_ENABLED
            // Load and build kernels
            ForwardKernel = CL.LoadBuildKernel(CL.KernelsPath + "/ReLUForward.cl", "ReLUForward");
            BackwardKernel = CL.LoadBuildKernel(CL.KernelsPath + "/ReLUBackward.cl", "ReLUBackward");
#endif
        }

        /// <summary>
        ///  Connect current layer to layer given as argument.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public override void ConnectTo(Layer PreviousLayer)
        {
            base.ConnectTo(PreviousLayer);

            this.nOutputUnits = PreviousLayer.Output.NumberOfUnits;
            this.outputNeurons = new Neurons(this.nOutputUnits);

        }

        /// <summary>
        /// Method to set this layer as the first layer of the network.
        /// </summary>
        public override void SetAsFirstLayer(int InputWidth, int InputHeight, int InputDepth)
        {
            throw new System.InvalidOperationException("You are setting a ReLU layer as first layer of the network...\nIs it really what you want to do?");
        }

        public override void InitializeParameters()
        {
#if OPENCL_ENABLED
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

            this.globalWorkSizePtr = new IntPtr[] { (IntPtr)(Output.NumberOfUnits) };
            int tmpLocalWorkSize = Output.NumberOfUnits;
            while (tmpLocalWorkSize > CL.MaxWorkGroupSize || tmpLocalWorkSize > CL.MaxWorkItemSizes[0])
                tmpLocalWorkSize /= 2;
            this.localWorkSizePtr = new IntPtr[] { (IntPtr)(tmpLocalWorkSize) };
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
            CL.Error |= Cl.SetKernelArg(ForwardKernel, 2, (IntPtr)sizeof(int), Output.NumberOfUnits);
            CL.CheckErr(CL.Error, "ReLU.FeedForward(): Cl.SetKernelArg");

            // Run kernel
            CL.Error = Cl.EnqueueNDRangeKernel( CL.Queue,
                                                ForwardKernel,
                                                1,
                                                null,
                                                globalWorkSizePtr,
                                                localWorkSizePtr,
                                                0,
                                                null,
                                                out CL.Event);
            CL.CheckErr(CL.Error, "ReLU.FeedForward(): Cl.EnqueueNDRangeKernel");

            CL.Error = Cl.Finish(CL.Queue);
            CL.CheckErr(CL.Error, "Cl.Finish");

            CL.Error = Cl.ReleaseEvent(CL.Event);
            CL.CheckErr(CL.Error, "Cl.ReleaseEvent");
#else
            float[] tmpOutput = new float[this.numberOfUnits];
            for (int i = 0; i < this.numberOfUnits; i++)
            {
                if (this.input.GetHost()[i] > 0)
                    tmpOutput[i] = this.input.GetHost()[i];
                else
                    tmpOutput[i] = 0.0F;
            }
            this.output.SetHost(tmpOutput);
#endif
        }



        public override void BackPropagate()
        {
#if OPENCL_ENABLED

            // Set kernel arguments
            CL.Error = Cl.SetKernelArg(BackwardKernel, 0, Input.DeltaGPU);
            CL.Error |= Cl.SetKernelArg(BackwardKernel, 1, Output.DeltaGPU);
            CL.Error |= Cl.SetKernelArg(BackwardKernel, 2, Input.ActivationsGPU);
            CL.Error |= Cl.SetKernelArg(BackwardKernel, 3, (IntPtr)sizeof(int), Input.NumberOfUnits);
            CL.CheckErr(CL.Error, "ReLU.BackPropagate(): Cl.SetKernelArg");

            // Run kernel
            CL.Error = Cl.EnqueueNDRangeKernel( CL.Queue,
                                                BackwardKernel,
                                                1,
                                                null,
                                                globalWorkSizePtr,
                                                localWorkSizePtr,
                                                0,
                                                null,
                                                out CL.Event);
            CL.CheckErr(CL.Error, "ReLU.BackPropagate(): Cl.EnqueueNDRangeKernel");

            CL.Error = Cl.Finish(CL.Queue);
            CL.CheckErr(CL.Error, "Cl.Finish");

            CL.Error = Cl.ReleaseEvent(CL.Event);
            CL.CheckErr(CL.Error, "Cl.ReleaseEvent");
#else
            for (int i = 0; i < this.numberOfUnits; i++)
                this.input.DeltaHost[i] = this.input.GetHost()[i] > 0 ? this.output.DeltaHost[i] : 0.0F;
#endif

        }

        #endregion


    }
}
