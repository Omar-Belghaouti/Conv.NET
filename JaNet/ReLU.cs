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
        #region Fields (private)

#if OPENCL_ENABLED
        private IntPtr[] globalWorkSizePtr;
        private IntPtr[] localWorkSizePtr; 
        // in this case nInput = nOutput  ==>  only need to set one global/local work size 
        // (i.e. no need to distinguish between forward and backward pass)

        private ErrorCode clError;
        private Event clEvent;
#endif

        #endregion


        #region Setup methods

        /// <summary>
        /// Constructor of ReLU layer.
        /// </summary>
        public ReLU()
        {
            this.type = "ReLU";
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

#if OPENCL_ENABLED
            this.clError = new ErrorCode();
            this.clEvent = new Event();
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
            while (tmpLocalWorkSize > OpenCLSpace.MaxWorkGroupSize || tmpLocalWorkSize > OpenCLSpace.MaxWorkItemSizes[0])
                tmpLocalWorkSize /= 2;
            this.localWorkSizePtr = new IntPtr[] { (IntPtr)(tmpLocalWorkSize) };
        }
#endif


        #endregion


        #region Operating methods

        public override void FeedForward()
        {
#if OPENCL_ENABLED
            // Set kernel arguments
            clError = Cl.SetKernelArg(OpenCLSpace.ReLUForward, 0, Output.ActivationsGPU);
            clError |= Cl.SetKernelArg(OpenCLSpace.ReLUForward, 1, Input.ActivationsGPU);
            clError |= Cl.SetKernelArg(OpenCLSpace.ReLUForward, 2, (IntPtr)sizeof(int), Output.NumberOfUnits);
            OpenCLSpace.CheckErr(clError, "ReLU.FeedForward(): Cl.SetKernelArg");

            // Run kernel
            clError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                OpenCLSpace.ReLUForward,
                                                1,
                                                null,
                                                globalWorkSizePtr,
                                                localWorkSizePtr,
                                                0,
                                                null,
                                                out clEvent);
            OpenCLSpace.CheckErr(clError, "ReLU.FeedForward(): Cl.EnqueueNDRangeKernel");

            clError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(clError, "Cl.Finish");

            clError = Cl.ReleaseEvent(clEvent);
            OpenCLSpace.CheckErr(clError, "Cl.ReleaseEvent");
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
            clError = Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 0, Input.DeltaGPU);
            clError |= Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 1, Output.DeltaGPU);
            clError |= Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 2, Input.ActivationsGPU);
            clError |= Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 3, (IntPtr)sizeof(int), Input.NumberOfUnits);
            OpenCLSpace.CheckErr(clError, "ReLU.BackPropagate(): Cl.SetKernelArg");

            // Run kernel
            clError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                OpenCLSpace.ReLUBackward,
                                                1,
                                                null,
                                                globalWorkSizePtr,
                                                localWorkSizePtr,
                                                0,
                                                null,
                                                out clEvent);
            OpenCLSpace.CheckErr(clError, "ReLU.BackPropagate(): Cl.EnqueueNDRangeKernel");

            //clError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(clError, "Cl.Finish");

            clError = Cl.ReleaseEvent(clEvent);
            OpenCLSpace.CheckErr(clError, "Cl.ReleaseEvent");
#else
            for (int i = 0; i < this.numberOfUnits; i++)
                this.input.DeltaHost[i] = this.input.GetHost()[i] > 0 ? this.output.DeltaHost[i] : 0.0F;
#endif

        }

        #endregion


    }
}
