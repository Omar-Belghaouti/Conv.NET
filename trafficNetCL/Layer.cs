using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    /// <summary>
    /// Base layer class. All other layers will inherit from this.
    /// </summary>
    public abstract class Layer
    {


        #region Layer class fields

        private object weights; // weight tensor, transforms 3D input to 3D output (should live on the GPU)
        private object biases; // same size as Output (should live on the GPU)
        private object deltas; // errors, to be computed with backprop (must have same size as Output)

        private Neurons input;
        private Neurons output;

        #endregion

        public virtual Neurons Input
        {
            get { return input; }
            set { input = value; }
        }

        public virtual Neurons Output
        {
            get { return output; }
        }

        // Alternative approach: link layer to the next like so:
        // public Layer NextLayer { get; set; }
        // And then in all loops over layer use this "pointer"... 
        // Q: But is this better, or even useful?
        // A: No, but we are going to NEED to use something like this if we want to create a multiscale network


        public abstract void ConnectTo(Layer PreviousLayer);


        /// <summary>
        /// Method to setup the first layer of the network.
        /// </summary>
        /// 
        public abstract void SetupAsFirstLayer(int InputImageWidth, int InputImageHeight, int InputImageDepth);

        /// <summary>
        /// Method to setup any layer in the network EXCEPT the first one.
        /// </summary>
        public abstract void Setup();

        public virtual void ForwardCPU()
        {
            
        }

        public virtual void ForwardGPU()
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
