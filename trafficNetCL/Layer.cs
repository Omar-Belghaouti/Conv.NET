using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
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

        protected int numberOfUnits;
        protected int id;
        protected string type;

        // Used in convolutional and pooling layers:
        protected int inputWidth;
        protected int inputHeight; // assumed equal to inputWidth for now
        protected int inputDepth;

        protected int outputWidth;
        protected int outputHeight; // assumed equal to outputWidth for now
        protected int outputDepth;

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

        public int ID
        {
            get { return id; }
            set { this.id = value; }
        }

        public string Type
        {
            get { return type; }
            set { this.type = value; }
        }

        // Used in convolutional and pooling layers:

        public int OutputWidth // allows setup of next layer
        {
            get { return outputWidth; }
        }

        public int OutputHeight // allows setup of next layer
        {
            get { return outputHeight; }
        }

        public int OutputDepth // allows setup of next layer
        {
            get { return outputDepth; }
        }
        #endregion


        #region Setup methods (public, to be called once)

        /// <summary>
        /// Set this as the first layer of the network.
        /// </summary>
        /// 
        public virtual void SetAsFirstLayer(int InputWidth, int InputHeight, int InputDepth)
        {

        }

        /// <summary>
        /// Connect this layer to previous one.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public virtual void ConnectTo(Layer PreviousLayer)
        {
            this.Input = PreviousLayer.Output; // assignment by reference! 
            // In memory, output neurons of previous layer and input neurons of current layer are the same thing!
        }


        /// <summary>
        /// Initialize layer parameters.
        /// </summary>
        public virtual void InitializeParameters()
        {
        }

        #endregion


        #region Training methods (public abstract)

        /// <summary>
        /// Run layer forward.
        /// </summary>
        public abstract void FeedForward();


        /// <summary>
        /// Compute errors with backpropagation.
        /// </summary>
        public abstract void BackPropagate();

        /// <summary>
        /// Updates layer parameters (if any)
        /// </summary>
        public virtual void UpdateParameters(double learningRate, double momentumMultiplier)
        {
        }

        /*
        public virtual void ClearDelta()
        {
            Array.Clear(this.input.DeltaHost, 0, this.input.NumberOfUnits);
            Array.Clear(this.output.DeltaHost, 0, this.output.NumberOfUnits);
        }
        */

        

        #endregion


        #region Debugging 
        
        public virtual void DisplayParameters()
        {

        }


        #endregion

    }
}
