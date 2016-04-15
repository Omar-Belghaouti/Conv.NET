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
