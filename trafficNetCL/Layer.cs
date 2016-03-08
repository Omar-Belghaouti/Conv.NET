using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace trafficNetCL
{
    /// <summary>
    /// Base layer class. All other layers will inherit from this.
    /// </summary>
    public class Layer : ILayer
    {
        private int InputWidth;
        private int InputHeight;
        private int InputDepth;

        private int OutputWidth;
        private int OutputHeight;
        private int OutputDepth;

        private float[, ,] Weights;
        private float[, ,] Biases;

        private float[, ,] Input;
        private float[, ,] Output;

        private float[, ,] Deltas;

        Layer NextLayer;

        public Layer(int inputW, int inputH, int inputD,
                     int outputW, int outputH, int outputD)
        {
            this.InputWidth = inputW;
            this.InputHeight = inputH;
            this.InputDepth = inputD;
            this.OutputWidth = outputW;
            this.OutputHeight = outputH;
            this.OutputDepth = outputD;
        }

        public virtual void ForwardBatch() 
        {

        }

        public virtual void ForwardOne()
        {

        }

        public virtual void BackProp()
        {

        }
    }
}
