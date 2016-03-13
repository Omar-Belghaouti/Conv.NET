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

        #region Layer base class fields (protected)

        protected Neurons input;
        protected Neurons output;
        protected Layer nextLayer;
        protected string layerType;

        #endregion


        #region Layer base class properties (public)

        public virtual Neurons Input
        {
            get { return input; }
            set { input = value; }
        }

        public virtual Neurons Output
        {
            get { return output; }
        }
        
        public virtual Layer NextLayer {
            set { this.nextLayer = value; } 
        }

        public string LayerType
        {
            get { return layerType; }
            set { this.layerType = value; }
        }

        #endregion


        #region Layer base class setup methods (public, to be called once)

        /// <summary>
        /// Link this layer to previous one.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public void ConnectTo(Layer PreviousLayer)
        {
            this.Input = PreviousLayer.Output;
        }


        /// <summary>
        /// Method to setup the first layer of the network.
        /// </summary>
        /// 
        public abstract void SetAsFirstLayer(int InputImageWidth, int InputImageHeight, int InputImageDepth);

        /// <summary>
        /// Initialize layer parameters (weights and biases).
        /// </summary>
        public abstract void InitializeParameters();

        #endregion


        #region Layer class methods (public abstract)

        /// <summary>
        /// Run layer forward (one input), using CPU
        /// </summary>
        public abstract void ForwardOneCPU();

        /// <summary>
        /// Run layer forward (one minibatch), using CPU
        /// </summary>
        public abstract void ForwardBatchCPU();

        /// <summary>
        /// Run layer forward using GPU
        /// </summary>
        public abstract void ForwardGPU();

        /// <summary>
        /// Compute errors with backpropagation (for one input/output) using CPU
        /// </summary>
        public abstract void BackPropOneCPU();

        /// <summary>
        /// Compute errors with backpropagation (for one minibatch) using CPU
        /// </summary>
        public abstract void BackPropBatchCPU();

        /// <summary>
        /// Compute errors with backpropagation using GPU
        /// </summary>
        public abstract void UpdateParameters();

        #endregion

    }
}
