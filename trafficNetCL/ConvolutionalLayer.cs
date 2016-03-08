using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace trafficNetCL
{
    class ConvolutionalLayer : Layer
    {
        public ConvolutionalLayer(int inputW, int inputH, int inputD, int outputW, int outputH, int outputD) 
                                : base(inputW, inputH, inputD, outputW, outputH, outputD)
        {
            // inherited constructor
        }


    }
}
