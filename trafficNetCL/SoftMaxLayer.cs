using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace TrafficNetCL
{
    class SoftMaxLayer : Layer
    {
        int NumberOfClasses;

        // Constructor
        public SoftMaxLayer(int nClasses)
        {
            Console.WriteLine("Adding a softmax layer with {0} output units...", nClasses);
            this.NumberOfClasses = nClasses;
        }

        public override void ConnectTo(Layer PreviousLayer)
        {

        }

        /// <summary>
        /// Method to setup the first layer of the network.
        /// </summary>
        public override void SetAsFirstLayer(int InputImageWidth, int InputImageHeight, int InputImageDepth)
        {
        }


        public override void InitializeParameters()
        {
        }

        public override void ForwardOneCPU()
        {

        }

        public override void BackPropOneCPU()
        {

        }

    }
}
