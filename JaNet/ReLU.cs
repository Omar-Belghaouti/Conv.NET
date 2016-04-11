﻿using System;
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
            //      global work size = smallest multiple of BASE_GROUP_SIZE larger than the total number of processes needed (for efficiency)
            //      local work size = largest multiple of BASE_GROUP_SIZE that global work size is a multiple of, with the constraint of being 
            //                          lesser or equal to current device's MaxWorkGroupSize and fwd/bwd kernels' maxKernelWorkGroupSize.
            // BASE_GROUP_SIZE is a constant, multiple of 2. Suggested values: 32 (Nvidia WARP) or 64 (AMD WAVEFRONT).

            int totalWorkItemsNeeded = OutputNeurons.NumberOfUnits;
            int smallestMultipleOfBGS = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(totalWorkItemsNeeded) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.globalWorkSizePtr = new IntPtr[] { (IntPtr)(smallestMultipleOfBGS) };

            int maxKernelWorkGroupSize = (int)Math.Max(Cl.GetKernelWorkGroupInfo(OpenCLSpace.ReLUForward,
                                                                                    OpenCLSpace.Device,
                                                                                    KernelWorkGroupInfo.WorkGroupSize,
                                                                                    out OpenCLSpace.ClError).CastTo<int>(),
                                                        Cl.GetKernelWorkGroupInfo(OpenCLSpace.ReLUBackward,
                                                                                    OpenCLSpace.Device,
                                                                                    KernelWorkGroupInfo.WorkGroupSize,
                                                                                    out OpenCLSpace.ClError).CastTo<int>());

            int localWorkSize = OpenCLSpace.BASE_GROUP_SIZE;
            while (localWorkSize <= OpenCLSpace.MaxWorkGroupSize && localWorkSize <= maxKernelWorkGroupSize)
            {
                int tmpLocalWorkSize = 2 * localWorkSize;
                if (smallestMultipleOfBGS % tmpLocalWorkSize == 0) // if global divides local
                    localWorkSize = tmpLocalWorkSize;
                else
                    break;
            }
            this.localWorkSizePtr = new IntPtr[] { (IntPtr)(localWorkSize) };
        }
#endif

        #endregion


        #region Training methods

        public override void FeedForward()
        {
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
#if OPENCL_ENABLED
                // Set kernel arguments
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ReLUForward, 0, OutputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ReLUForward, 1, InputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ReLUForward, 2, (IntPtr)sizeof(int), OutputNeurons.NumberOfUnits);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ReLU.FeedForward(): Cl.SetKernelArg");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
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
                float[] tmpOutput = new float[this.nOutputUnits];
                for (int i = 0; i < this.nOutputUnits; i++)
                {
                    if (this.inputNeurons.GetHost()[m][i] > 0)
                        tmpOutput[i] = this.inputNeurons.GetHost()[m][i];
                    else
                        tmpOutput[i] = 0.0F;
                }
                this.outputNeurons.SetHost(m, tmpOutput);
#endif
            }
        }



        public override void BackPropagate()
        {
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
#if OPENCL_ENABLED
                // Set kernel arguments
                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 0, InputNeurons.DeltaGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 1, OutputNeurons.DeltaGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 2, InputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ReLUBackward, 3, (IntPtr)sizeof(int), InputNeurons.NumberOfUnits);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ReLU.BackPropagate(): Cl.SetKernelArg");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
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
                for (int i = 0; i < this.nOutputUnits; i++)
                    this.inputNeurons.DeltaHost[m][i] = this.inputNeurons.GetHost()[m][i] > 0 ? this.outputNeurons.DeltaHost[m][i] : 0.0F;
#endif
            }
        }

        #endregion


    }
}