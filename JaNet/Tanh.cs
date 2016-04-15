using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    
    class Tanh : Layer
    {
        #region Tanh layer class fields (private)

        private double beta;

#if OPENCL_ENABLED

        private IntPtr[] globalWorkSizePtr;
        private IntPtr[] localWorkSizePtr; 
        // in this case nInput = nOutput  ==>  only need to set one global/local work size 
        // (i.e. no need to distinguish between forward and backward pass)
#endif

        #endregion


        #region Setup methods (to be called once)

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

        public override void ConnectTo(Layer PreviousLayer)
        {
            base.ConnectTo(PreviousLayer);

            this.nOutputUnits = nInputUnits;
            this.outputNeurons = new Neurons(nOutputUnits);

            this.outputWidth = inputWidth;
            this.outputHeight = inputHeight;
            this.outputDepth = inputDepth;

#if OPENCL_ENABLED
            SetWorkGroupSizes();
#endif
        }

#if OPENCL_ENABLED
        private void SetWorkGroupSizes()
        {
            // TODO: 
            
            // Work group sizes will be set as follows:
            //      global work size = smallest multiple of BASE_GROUP_SIZE larger than the total number of processes needed (for efficiency)
            //      local work size = largest multiple of BASE_GROUP_SIZE that global work size is a multiple of, with the constraint of being 
            //                          lesser or equal to current device's MaxWorkGroupSize and fwd/bwd kernels' maxKernelWorkGroupSize.
            // BASE_GROUP_SIZE is a constant, multiple of 2. Suggested values: 32 (Nvidia WARP) or 64 (AMD WAVEFRONT).

            int totalWorkItemsNeeded = OutputNeurons.NumberOfUnits;
            int smallestMultipleOfBGS = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(totalWorkItemsNeeded) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.globalWorkSizePtr = new IntPtr[] { (IntPtr)(smallestMultipleOfBGS) };

            int maxKernelWorkGroupSize = (int)Math.Max(Cl.GetKernelWorkGroupInfo(OpenCLSpace.TanhForward,
                                                                                    OpenCLSpace.Device,
                                                                                    KernelWorkGroupInfo.WorkGroupSize,
                                                                                    out OpenCLSpace.ClError).CastTo<int>(),
                                                        Cl.GetKernelWorkGroupInfo(  OpenCLSpace.TanhBackward,
                                                                                    OpenCLSpace.Device,
                                                                                    KernelWorkGroupInfo.WorkGroupSize,
                                                                                    out OpenCLSpace.ClError).CastTo<int>());

            int localWorkSize = OpenCLSpace.BASE_GROUP_SIZE;
            while (true)
            {
                int tmpLocalWorkSize = 2 * localWorkSize;

                bool globalDividesLocal = smallestMultipleOfBGS % tmpLocalWorkSize == 0;
                bool isLocalGroupTooLarge = tmpLocalWorkSize > OpenCLSpace.MaxWorkGroupSize || tmpLocalWorkSize > maxKernelWorkGroupSize;

                if (globalDividesLocal && !isLocalGroupTooLarge) // if global divides local and it's not too large
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
                OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.TanhForward, 0, OutputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhForward, 1, InputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhForward, 2, (IntPtr)sizeof(float), beta);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhForward, 3, (IntPtr)sizeof(int), OutputNeurons.NumberOfUnits);
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
#else
                double[] tmpOutput = new double[nOutputUnits];
                for (int i = 0; i < nOutputUnits; i++)
                {
                    tmpOutput[i] = (float)Math.Tanh(beta * this.inputNeurons.GetHost()[m][i]);
                }
                outputNeurons.SetHost(m, tmpOutput);
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
                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.TanhBackward, 0, inputNeurons.DeltaGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhBackward, 1, outputNeurons.DeltaGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhBackward, 2, outputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhBackward, 3, (IntPtr)sizeof(float), beta);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.TanhBackward, 3, (IntPtr)sizeof(int), inputNeurons.NumberOfUnits);
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
#else
                for (int i = 0; i < nInputUnits; i++)
                {
                    double derivative = beta * (1 - Math.Pow(outputNeurons.GetHost()[m][i], 2));
                    inputNeurons.DeltaHost[m][i] = this.outputNeurons.DeltaHost[m][i] * derivative;
                }
#endif
            }

#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#endif









        }


        

        #endregion


    }

}
