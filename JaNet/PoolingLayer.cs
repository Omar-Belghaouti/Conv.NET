using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JaNet
{
    class Pooling : Layer
    {
        #region Fields

        private string poolingType; // options: "max/Max", "average/Average", "l2norm/L2norm"
        private int poolWidth;
        private int stride;

#if OPENCL_ENABLED
        private IntPtr[] globalWorkSizePtr;
        private IntPtr[] localWorkSizePtr;
#endif

        #endregion


        #region Setup methods

        /// <summary>
        /// Constructor of Pooling layer
        /// </summary>
        /// <param name="PoolingType"></param>
        /// <param name="poolWidth"></param>
        /// <param name="stride"></param>
        public Pooling(string PoolingType, int PoolWidth, int Stride)
        {
            this.type = "Pooling";
            this.poolWidth = PoolWidth;
            this.stride = Stride;
        }

        public override void SetupOutput()
        {
            if (poolWidth == stride && inputWidth % poolWidth != 0)
                throw new ArgumentException("Cannot apply pooling to input: pooling width and stride do not fit!");

            this.outputWidth = (inputWidth - poolWidth) / stride + 1;
            this.outputHeight = (inputHeight - poolWidth) / stride + 1;
            this.outputDepth = inputDepth;

            this.nOutputUnits = outputWidth * outputHeight * outputDepth;
            this.outputNeurons = new Neurons(nOutputUnits);
        }


        public override void SetWorkGroups()
        {

            // TODO: method

            /*
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
            int totalWorkItemsNeeded = nOutputUnits * outputNeurons.MiniBatchSize;
            int smallestMultipleOfLocal = (int)(OpenCLSpace.OPTIMAL_GROUP_SIZE * Math.Ceiling((double)(totalWorkItemsNeeded) / (double)OpenCLSpace.OPTIMAL_GROUP_SIZE));
            this.globalWorkSizePtr = new IntPtr[] { (IntPtr)(smallestMultipleOfLocal) };
#endif
             * */
        }


        #endregion


#region Methods

        public override void FeedForward()
        {
#if OPENCL_ENABLED

#else
            //TODO: CPU code
#endif

        }

        public override void BackPropagate()
        {
#if OPENCL_ENABLED

#else
            //TODO: CPU code
#endif
        }

#endregion



    }
}
