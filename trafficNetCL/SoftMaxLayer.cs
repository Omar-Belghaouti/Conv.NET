using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


namespace trafficNetCL
{
    class SoftMaxLayer : Layer, ILayer
    {
        int NumberOfClasses;

        // Constructor
        public SoftMaxLayer(int nClasses)
        {
            Console.WriteLine("Adding a softmax layer with {0} output units...", nClasses);
            this.NumberOfClasses = nClasses;
        }

        public override void SetupOutput()
        {
            this.OutputWidth = 1;
            this.OutputHeight = this.NumberOfClasses;
            this.OutputDepth = 1;
            this.Output = new float[1, this.NumberOfClasses, 1];
        }

    }
}
