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

        public override void InitializeParameters()
        {
        }

        /// <summary>
        /// Method to setup the first layer of the network.
        /// </summary>
        public override void SetAsFirstLayer(int[] InputDimensions)
        {
        }

        public override void ForwardOneCPU()
        {
        }

        public override void ForwardBatchCPU()
        {
        }

        public override void ForwardGPU()
        {
        }

        public override void BackPropOneCPU()
        {

        }

        public override void BackPropOneCPU(float[] costGradient)
        {
            throw new System.InvalidOperationException("You are using a convolutional layer as output layer of the network...\nIs it really what you want to do?");
        }

        public override void BackPropBatchCPU()
        {
        }

        public override void UpdateParameters(double learningRate)
        {
        }


    }
}
