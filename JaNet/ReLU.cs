using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    // CLEAN

    class ReLU : Layer
    {
        #region Fields

#if OPENCL_ENABLED

        private IntPtr[] globalWorkSizePtr;
        private IntPtr[] localWorkSizePtr; 
        // in this case nInput = nOutput  ==>  only need to set one global/local work size 
        // (i.e. no need to distinguish between forward and backward pass)
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

            this.nOutputUnits = PreviousLayer.OutputNeurons.NumberOfUnits;
            this.outputNeurons = new Neurons(this.nOutputUnits);
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

            this.globalWorkSizePtr = new IntPtr[] { (IntPtr)(OutputNeurons.NumberOfUnits) };
            int tmpLocalWorkSize = OutputNeurons.NumberOfUnits;
            while (tmpLocalWorkSize > OpenCLSpace.MaxWorkGroupSize || tmpLocalWorkSize > OpenCLSpace.MaxWorkItemSizes[0])
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
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ReLUForward, 0, OutputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ReLUForward, 1, InputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ReLUForward, 2, (IntPtr)sizeof(int), OutputNeurons.NumberOfUnits);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ReLU.FeedForward(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel( OpenCLSpace.Queue,
                                                OpenCLSpace.ReLUForward,
                                                1,
                                                null,
                                                globalWorkSizePtr,
                                                localWorkSizePtr,
                                                0,
                                                null,
                                                out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ReLU.FeedForward(): Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
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
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 0, InputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 1, OutputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 2, InputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 3, (IntPtr)sizeof(int), InputNeurons.NumberOfUnits);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ReLU.BackPropagate(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel( OpenCLSpace.Queue,
                                                OpenCLSpace.ReLUBackward,
                                                1,
                                                null,
                                                globalWorkSizePtr,
                                                localWorkSizePtr,
                                                0,
                                                null,
                                                out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ReLU.BackPropagate(): Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
#else
            for (int i = 0; i < this.numberOfUnits; i++)
                this.input.DeltaHost[i] = this.input.GetHost()[i] > 0 ? this.output.DeltaHost[i] : 0.0F;
#endif

        }

        #endregion


    }
}
