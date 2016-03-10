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

        /// <summary>
        /// Method to setup the first layer of the network.
        /// </summary>
        public override void Setup(int InputImageWidth, int InputImageHeight, int InputImageDepth)
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
