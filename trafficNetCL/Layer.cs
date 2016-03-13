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
        private float[] deltas; // errors, to be computed with backprop (must have same size as Output)

        protected Neurons input;
        protected Neurons output;

        protected Layer nextLayer;

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

        public virtual float[] Delta
        {
            get { return this.deltas; }
        }

        // Alternative approach: link layer to the next like so:
        public virtual Layer NextLayer {
            set { this.nextLayer = value; } 
        }
        // And then in all loops over layer use this "pointer"... 
        // Q: But is this better, or even useful?
        // A: No, but we are going to NEED to use something like this if we want to create a multiscale network


        public abstract void ConnectTo(Layer PreviousLayer);


        /// <summary>
        /// Method to setup the first layer of the network.
        /// </summary>
        /// 
        public abstract void SetAsFirstLayer(int InputImageWidth, int InputImageHeight, int InputImageDepth);

        /// <summary>
        /// Initialize layer parameters (weights and biases).
        /// </summary>
        public abstract void InitializeParameters();

        public abstract void ForwardOneCPU();

        public virtual void ForwardGPU()
        {
            // here we'll use GPU
        }

        public abstract void BackPropOneCPU();

        public virtual void UpdateWeights()
        {

        }
    }
}
