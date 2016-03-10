using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    class ConvolutionalLayer : Layer
    {
        int FilterSize;

        // TO-DO: implement constructor
        public ConvolutionalLayer(int filterSize, int numFilters)
        {

        }

        // TO-DO: implement convolutional layer methods

        public override void ConnectTo(Layer PreviousLayer)
        {

        }

        /// <summary>
        /// Method to setup the first layer of the network.
        /// </summary>
        public override void SetupAsFirstLayer(int InputImageWidth, int InputImageHeight, int InputImageDepth)
        {
        }

        /// <summary>
        /// Method to setup any layer in the network EXCEPT the first one.
        /// </summary>
        public override void Setup()
        {
        }


    }
}
