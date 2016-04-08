using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JaNet
{
    /// <summary>
    /// This is a just a dummy layer to be set as the 0th layer of the network.
    /// It only contains output Neurons, which will be used to feed the data into the next layer.
    /// </summary>
    class InputLayer : Layer
    {
        /// <summary>
        /// InputLayer class standard constructor.
        /// </summary>
        /// <param name="MiniBatchSize"></param>
        /// <param name="DataDepth"></param>
        /// <param name="DataHeight"></param>
        /// <param name="DataWidth"></param>
        public InputLayer(int MiniBatchSize, int DataDepth, int DataHeight, int DataWidth)
        {
            this.type = "Input";

            // TODO: remove this after implementing mini-batch training
            if (MiniBatchSize != 1)
                throw new ArgumentException("Mini-batch training not implemented yet. Use MiniBatchSize = 1.");
            this.nOutputUnits = MiniBatchSize * DataDepth * DataHeight * DataWidth;

            this.outputDepth = DataDepth;
            this.outputHeight = DataHeight; // assumed equal to outputWidth
            this.outputWidth = DataWidth;

            this.outputNeurons = new Neurons(this.nOutputUnits);
        }

        /// <summary>
        /// Feeds data to the next layer (1st layer of the network).
        /// TODO: implement mini-batch training.
        /// </summary>
        /// <param name="dataSet"></param>
        /// <param name="iDataPoint"></param>
        public override void Feed(DataSet dataSet, int iDataPoint)
        {
#if OPENCL_ENABLED
            this.outputNeurons.ActivationsGPU = dataSet.DataGPU(iDataPoint);
#else
            this.outputNeurons.SetHost(dataSet.GetDataPoint(iDataPoint));
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
