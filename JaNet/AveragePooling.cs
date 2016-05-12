using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    [Serializable]
    class AveragePooling : Layer
    {
        #region Fields

        private int inputArea;

        private IntPtr[] fwdGlobalWorkSizePtr;
        private IntPtr[] fwdLocalWorkSizePtr;

        private IntPtr[] bwdGlobalWorkSizePtr;
        private IntPtr[] bwdLocalWorkSizePtr;

        #endregion


        #region Setup methods

        /// <summary>
        /// Constructor of MaxPooling layer
        /// </summary>
        /// <param name="poolWidth"></param>
        /// <param name="stride"></param>
        public AveragePooling()
        {
            this.type = "AveragePooling";

        }

        public override void SetupOutput()
        {
            this.inputArea = inputHeight * inputWidth; 

            // Setup output __________________________________________________________________________________________

            // Currently only supporting global pooling, i.e. average over all spatial dimensions
            this.outputWidth = 1;
            this.outputHeight = 1;
            this.outputDepth = inputDepth;

            this.nOutputUnits = outputDepth;
            this.outputNeurons = new Neurons(nOutputUnits);

        }


        public override void SetWorkGroups()
        {

            // TODO: method

            
#if OPENCL_ENABLED
            // Work group sizes will be set as follows:
            //      global work size = smallest multiple of OPTIMAL_GROUP_SIZE larger than 
            //                         the total number of processes needed (for efficiency).
            //      local work size = as close as possible to OPTIMAL_GROUP_SIZE (making sure 
            //                        that global worksize is a multiple of this)
            // OPTIMAL_GROUP_SIZE is a small multiple of BASE_GROUP_SIZE, which in turn is a 
            //                    constant multiple of 2, platform-dependent, e.g. 32 (Nvidia 
            //                    WARP) or 64 (AMD WAVEFRONT).

            // Forward Local
            int optBaseRatio = OpenCLSpace.OPTIMAL_GROUP_SIZE / OpenCLSpace.BASE_GROUP_SIZE;
            this.fwdLocalWorkSizePtr = new IntPtr[] { (IntPtr)optBaseRatio, (IntPtr)OpenCLSpace.BASE_GROUP_SIZE };

            // Forward Global
            int smallestMultiple0 = (int)(optBaseRatio * Math.Ceiling((double)(inputNeurons.MiniBatchSize) / (double)optBaseRatio));
            int smallestMultiple1 = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(inputDepth) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.fwdGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple0, (IntPtr)smallestMultiple1 };

            // Backward Local
            this.bwdLocalWorkSizePtr = new IntPtr[] { (IntPtr)optBaseRatio, (IntPtr)OpenCLSpace.BASE_GROUP_SIZE };

            // Backward Global
            int smallestMultiple1bwd = (int)(OpenCLSpace.BASE_GROUP_SIZE * Math.Ceiling((double)(nInputUnits) / (double)OpenCLSpace.BASE_GROUP_SIZE));
            this.bwdGlobalWorkSizePtr = new IntPtr[] { (IntPtr)smallestMultiple0, (IntPtr)smallestMultiple1bwd };
#endif
             
        }


        #endregion


#region Methods

        public override void FeedForward()
        {
#if TIMING_LAYERS
            // TODO: add timer
#endif

            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.AveragePoolingForward, 0, outputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.AveragePoolingForward, 1, inputNeurons.ActivationsGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.AveragePoolingForward, 2, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.AveragePoolingForward, 3, (IntPtr)sizeof(int), inputArea);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.AveragePoolingForward, 4, (IntPtr)sizeof(int), inputDepth);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.AveragePoolingForward, 5, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.AveragePoolingForward,
                                                            2,
                                                            null,
                                                            fwdGlobalWorkSizePtr,
                                                            fwdLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

#if TIMING_LAYERS
            // TODO: add timer
#endif
        }

        public override void BackPropagate()
        {
#if TIMING_LAYERS
            // TODO: add timer
#endif

            OpenCLSpace.ClError  = Cl.SetKernelArg(OpenCLSpace.AveragePoolingBackward, 0, inputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.AveragePoolingBackward, 1, outputNeurons.DeltaGPU);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.AveragePoolingBackward, 2, (IntPtr)sizeof(int), nInputUnits);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.AveragePoolingBackward, 3, (IntPtr)sizeof(int), inputArea);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.AveragePoolingBackward, 4, (IntPtr)sizeof(int), inputDepth);
            OpenCLSpace.ClError |= Cl.SetKernelArg(OpenCLSpace.AveragePoolingBackward, 5, (IntPtr)sizeof(int), inputNeurons.MiniBatchSize);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.SetKernelArg");

            OpenCLSpace.ClError = Cl.EnqueueNDRangeKernel(  OpenCLSpace.Queue,
                                                            OpenCLSpace.AveragePoolingBackward,
                                                            2,
                                                            null,
                                                            bwdGlobalWorkSizePtr,
                                                            bwdLocalWorkSizePtr,
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueNDRangeKernel");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");


#if TIMING_LAYERS
            // TODO: add timer
#endif
        }

#endregion



    }
}
