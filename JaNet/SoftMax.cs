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
        #region Fields

#if OPENCL_ENABLED
        private Mem auxiliaryFloatBuffer; // needed by forward pass

        private IntPtr[] globalWorkSizePtr;
        private IntPtr[] localWorkSizePtr;
        // in this case nInput = nOutput  ==>  only need to set one global/local work size 
        // (i.e. no need to distinguish between forward and backward pass)
#endif

        #endregion


        #region Setup methods

        /// <summary>
        /// Constructor of Softmax layer.
        /// </summary>
        /// <param name="Beta"></param>
        public SoftMax()
        {
            this.type = "SoftMax";
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
            this.auxiliaryFloatBuffer = (Mem)Cl.CreateBuffer(   OpenCLSpace.Context, 
                                                                MemFlags.ReadWrite, 
                                                                (IntPtr)sizeof(float), 
                                                                out OpenCLSpace.ClError);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.CreateBuffer auxiliaryFloatBuffer");

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


        #region Methods

        public override void FeedForward()
        {
            for (int m = 0; m < inputNeurons.MiniBatchSize; m++)
            {
#if OPENCL_ENABLED
                // Set kernel arguments
                OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.SoftmaxForward, 0, OutputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.SoftmaxForward, 1, InputNeurons.ActivationsGPU[m]);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.SoftmaxForward, 2, auxiliaryFloatBuffer);
                OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.SoftmaxForward, 3, (IntPtr)sizeof(int), OutputNeurons.NumberOfUnits);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Softmax.FeedForward(): Cl.SetKernelArg");

                // Run kernel
                OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(OpenCLSpace.Queue,
                                                    OpenCLSpace.SoftmaxForward,
                                                    1,
                                                    null,
                                                    globalWorkSizePtr,
                                                    localWorkSizePtr,
                                                    0,
                                                    null,
                                                    out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Softmax.FeedForward(): Cl.EnqueueNDRangeKernel");

                OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
#else

                // use rescaling trick to improve numerical stability
                float maxInput = this.inputNeurons.GetHost()[m][0];
                for (int i = 1; i < nOutputUnits; i++)
                {
                    if (this.inputNeurons.GetHost()[m][i] > maxInput)
                        maxInput = this.inputNeurons.GetHost()[m][i];
                }

                float[] tmpOutput = new float[nOutputUnits];
                for (int i = 0; i < nOutputUnits; i++)
                {
                    tmpOutput[i] = (float)Math.Exp(this.inputNeurons.GetHost()[m][i] - maxInput);
                }
                float sum = tmpOutput.Sum();
                for (int i = 0; i < nOutputUnits; i++)
                {
                    tmpOutput[i] /= sum;
                }

                this.outputNeurons.SetHost(m, tmpOutput);
#endif
            }
        }


        public override void BackPropagate()
        {
            throw new System.InvalidOperationException("Called BackPropagate() method of SoftMax layer. Don't do it! Just feed the gradient back to the previous layer!");
            // NO backprop here!!
            // Compute directly input.Delta from cross-entropy cost: faster and numerically more stable
        }

        #endregion


    }
}
