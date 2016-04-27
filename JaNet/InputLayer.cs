using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    /// <summary>
    /// This is a just a dummy layer to be set as the 0th layer of the network.
    /// It only contains output Neurons, which will be used to feed the data into the next layer.
    /// </summary>
    class InputLayer : Layer
    {
        #region Fields

        private int imageChannels;
        private int imageHeight;
        private int imageWidth;

        #endregion


        /// <summary>
        /// InputLayer class standard constructor.
        /// </summary>
        /// <param name="DataDepth"></param>
        /// <param name="DataHeight"></param>
        /// <param name="DataWidth"></param>
        public InputLayer(int DataDepth, int DataHeight, int DataWidth)
        {
            this.type = "Input";

            if (DataHeight != DataWidth)
                throw new ArgumentException("Non-square input images are currently not supported.");

            this.imageChannels = DataDepth;
            this.imageHeight = DataHeight;
            this.imageWidth = DataWidth;

        }

        public override void SetupOutput()
        {
            this.outputDepth = imageChannels;
            this.outputHeight = imageHeight;
            this.outputWidth = imageWidth;

            this.nOutputUnits = outputDepth * outputHeight * outputWidth;

            this.outputNeurons = new Neurons(this.nOutputUnits);
        }

        public void FeedData(DataSet dataSet, int[] iExamples)
        {

#if TIMING_LAYERS
            Utils.InputFeedTimer.Start();
#endif
            int dataPointSize = dataSet.DataDimension;

            for (int m = 0; m < outputNeurons.MiniBatchSize; m++)
            {
#if OPENCL_ENABLED
                int iDataPoint = iExamples[m];

                OpenCLSpace.ClError = Cl.EnqueueCopyBuffer(OpenCLSpace.Queue,
                                                            dataSet.DataGPU[iDataPoint], // source
                                                            outputNeurons.ActivationsGPU, // destination
                                                            (IntPtr)0, // source offset (in bytes)
                                                            (IntPtr)(sizeof(float) * m * dataPointSize), // destination offset (in bytes)
                                                            (IntPtr)(sizeof(float) * dataPointSize),  // size of buffer to copy
                                                            0,
                                                            null,
                                                            out OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "InputLayer.FeedData Cl.EnqueueCopyBuffer inputData");

                OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");
#else
                outputNeurons.SetHost(m, dataSet.Data[iExamples[m]]);
#endif
            }

#if OPENCL_ENABLED
            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
#endif

#if TIMING_LAYERS
            Utils.InputFeedTimer.Stop();
#endif
        }

        public override void FeedForward()
        {
            throw new NotImplementedException("The network's input layer should not be run forward.");
        }

        public override void BackPropagate()
        {
            throw new NotImplementedException("The network's input layer should not be run backwards.");
        }


    }
}
