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

        // Fields and properties
        float[, , ,] Weights; // weight tensor, transforms 3D input to 3D output (should live on the GPU)
        float[, ,] Biases; // same size as Output (should live on the GPU)
        float[, ,] Deltas; // errors, to be computed with backprop 
                           // must have same size as Input (or Output?)

        public int InputWidth { get; set; }
        public int InputHeight { get; set; }
        public int InputDepth { get; set; }

        public int OutputWidth { get; set; }
        public int OutputHeight { get; set; }
        public int OutputDepth { get; set; }

        public float[, ,] Input { get; set; } // (should live on the GPU)
        public float[, ,] Output { get; set; } // (should live on the GPU)


        // Alternative approach: link layer to the next like so:
        public Layer NextLayer { get; set; }
        // And then in all loops over layer use this "pointer"... but is this better?

        public void SetupInput(int inputW, int inputH, int inputD)
        {
            // initialize 3D input array
            this.InputWidth = inputW;
            this.InputHeight = inputH;
            this.InputDepth = inputD;
            this.Input = new float[inputW, inputH, inputD];
        }

        public virtual void SetupOutput() 
        {
            // initialize output array based on input dimension and layer properties (depends on type of layer)
        }

        public virtual void SetupOutput(int nOutputClasses) // override this in SoftMaxLayer class
        {
            Console.WriteLine("Error: tried to call method 'SetupOutput(int nOutputClasses)', but this is not the output layer. ");
            Console.ReadKey();
            Environment.Exit(0);
        }

        public virtual void InitializeWeightsAndBiases()
        {

        }

        public virtual void ForwardBatch() 
        {
            // here we'll use GPU
        }

        public virtual void ForwardOne()
        {
            // here we'll use GPU
        }

        public virtual void BackProp()
        {
            // here we'll use GPU
        }

        public virtual void UpdateWeights()
        {

        }
    }
}
