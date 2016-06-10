using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    [Serializable]
    public class Tanh : Layer
    {
        #region Fields

        private double beta;

#if OPENCL_ENABLED

        private IntPtr[] globalWorkSizePtr;
        private IntPtr[] localWorkSizePtr; 
        // in this case nInput = nOutput  ==>  only need to set one global/local work size 
        // (i.e. no need to distinguish between forward and backward pass)
#endif

        #endregion


        #region Setup methods

        /// <summary>
        /// Constructor of Tanh layer. Specify beta parameter as argument.
        /// </summary>
        /// <param name="Beta"></param>
        public Tanh(double Beta)
        {
            //Console.WriteLine("Adding a tanh layer with activation parameter {0}...", Beta);

            this.beta = Beta;
            this.type = "Tanh";
        }

        public override void SetupOutput()
        {
            this.outputWidth = inputWidth;
            this.outputHeight = inputHeight;
            this.outputDepth = inputDepth;

            this.nOutputUnits = nInputUnits;
            this.outputNeurons = new Neurons(nOutputUnits);
        }


        public override void SetWorkGroups()
        {
#if OPENCL_ENABLED

            this.localWorkSizePtr = new IntPtr[] { (IntPtr)(OpenCLSpace.OPTIMAL_GROUP_SIZE) };

            int totalWorkItemsNeeded = OutputNeurons.NumberOfUnits * inputNeurons.MiniBatchSize;
            int smallestMultiple = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE * Math.Ceiling((double)(totalWorkItemsNeeded) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
            this.globalWorkSizePtr = new IntPtr[] { (IntPtr)(smallestMultiple) };
#endif
        }


        #endregion


        #region Methods

        public override void FeedForward()
        {
            // Set kernel arguments
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.TanhForward, 0, OutputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhForward, 1, InputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhForward, 2, (IntPtr)sizeof(float), beta);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhForward, 3, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhForward, 4, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Tanh.FeedForward(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.TanhForward,
                                                            1,
                                                            null,
                                                            globalWorkSizePtr,
                                                            localWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Tanh.FeedForward(): Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");


            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

        }

        public override void BackPropagate()
        {
           
            // Set kernel arguments
            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.TanhBackward, 0, inputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhBackward, 1, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhBackward, 2, outputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhBackward, 3, (IntPtr)sizeof(float), beta);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhBackward, 4, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhBackward, 5, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Tanh.BackPropagate(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.TanhBackward,
                                                            1,
                                                            null,
                                                            globalWorkSizePtr,
                                                            localWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Tanh.BackPropagate(): Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
        }

        #endregion

    }
}
