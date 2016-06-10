using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    [Serializable]
    public class ELU : Layer
    {
        #region Fields

        float alpha;

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
        public ELU(float Alpha)
        {
            this.type = "ELU";

            this.alpha = Alpha;
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
            // Work group sizes will be set as follows:
            //      global work size = smallest multiple of OPTIMAL_GROUP_SIZE larger than 
            //                         the total number of processes needed (for efficiency).
            //      local work size = as close as possible to OPTIMAL_GROUP_SIZE (making sure 
            //                        that global worksize is a multiple of this)
            // OPTIMAL_GROUP_SIZE is a small multiple of BASE_GROUP_SIZE, which in turn is a 
            //                    constant multiple of 2, platform-dependent, e.g. 32 (Nvidia 
            //                    WARP) or 64 (AMD WAVEFRONT).

            // Local
            this.localWorkSizePtr = new IntPtr[] { (IntPtr)OpenCLSpace.OPTIMAL_GROUP_SIZE };

            // Global
            int smallestMultipleOfLocal = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE * Math.Ceiling((double)(nOutputUnits * outputNeurons.MiniBatchSize) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
            this.globalWorkSizePtr = new IntPtr[] { (IntPtr)(smallestMultipleOfLocal) };
#endif
        }


        #endregion


        #region Methods

        public override void FeedForward()
        {
#if TIMING_LAYERS
            Utils.NonlinearityForwardTimer.Start();
#endif

#if OPENCL_ENABLED
            // Set kernel arguments
            OpenCLSpace.ClError = Cl.SetKernelArg(OpenCLSpace.ELUForward, 0, OutputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ELUForward, 1, InputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ELUForward, 2, (IntPtr)sizeof(float), alpha);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ELUForward, 3, (IntPtr)sizeof(int), OutputNeurons.NumberOfUnits * inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ELU.FeedForward(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.ELUForward,
                                                            1,
                                                            null,
                                                            globalWorkSizePtr,
                                                            localWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ELU.FeedForward(): Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {

                double[] tmpOutput = new double[this.nOutputUnits];
                for (int i = 0; i < this.nOutputUnits; i++)
                {
                    if (this.inputNeurons.GetHost()[m][i] > 0)
                        tmpOutput[i] = this.inputNeurons.GetHost()[m][i];
                    else
                        tmpOutput[i] = alpha * (Math.Exp(this.inputNeurons.GetHost()[m][i]) - 1.0 );
                }
                this.outputNeurons.SetHost(m, tmpOutput);

            }
#endif

#if TIMING_LAYERS
            Utils.NonlinearityForwardTimer.Stop();
#endif
        }



        public override void BackPropagate()
        {

#if TIMING_LAYERS
            Utils.NonlinearityBackpropTimer.Start();
#endif

#if OPENCL_ENABLED
            // Set kernel arguments
            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.ELUBackward, 0, inputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ELUBackward, 1, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ELUBackward, 2, inputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ELUBackward, 3, (IntPtr)sizeof(float), alpha);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.ELUBackward, 4, (IntPtr)sizeof(int), nInputUnits * inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ELU.BackPropagate(): Cl.SetKernelArg");

            // Run kernel
            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                            OpenCLSpace.ELUBackward,
                                                            1,
                                                            null,
                                                            globalWorkSizePtr,
                                                            localWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "ELU.BackPropagate(): Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#else
            throw new NotImplementedException("CPU code for ELUs not implemented yet.");
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
                for (int i = 0; i < nOutputUnits; i++)
                    //inputNeurons.DeltaHost[m][i] = inputNeurons.GetHost()[m][i] > 0 ? outputNeurons.DeltaHost[m][i] : 0.0;

            }
#endif

#if TIMING_LAYERS
            Utils.NonlinearityBackpropTimer.Stop();
#endif
        }

        #endregion


    }
}

