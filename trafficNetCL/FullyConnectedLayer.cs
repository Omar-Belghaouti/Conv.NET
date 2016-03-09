using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace trafficNetCL
{
    class FullyConnectedLayer : Layer, ILayer
    {
        int NumberOfUnits;

        // Constructor
        public FullyConnectedLayer(int nUnits)
        {
            Console.WriteLine("Adding a fully connected layer with {0} units...", nUnits);
            this.NumberOfUnits = nUnits;
        }

        public override void SetupOutput()
        {
            this.OutputWidth = 1;
            this.OutputHeight = NumberOfUnits;
            this.OutputDepth = 1;
            this.Output = new float[OutputWidth, OutputHeight, OutputDepth];
        }



    }
}
