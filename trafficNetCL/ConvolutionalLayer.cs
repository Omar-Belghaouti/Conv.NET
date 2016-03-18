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
        public ConvolutionalLayer(int filterSize, int numFilters, int strideLength)
        {
            Console.WriteLine("Adding a convolutional layer with {0} filters of size {1} and stride length {2}...", 
                filterSize, numFilters, strideLength);

            this.layerType = "Convolutional";
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

        public override void BackPropBatchCPU()
        {
        }

        public override void UpdateParameters(double learningRate, double momentumCoefficient)
        {
        }


    }
}
