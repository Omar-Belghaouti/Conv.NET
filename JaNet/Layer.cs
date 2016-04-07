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

        protected string type;
        protected int id;

        protected int nInputUnits;
        protected int nOutputUnits;

        protected Neurons inputNeurons;
        protected Neurons outputNeurons;

        protected int inputDepth;  
        protected int inputHeight; // assumed equal to inputWidth
        protected int inputWidth;

        protected int outputDepth;
        protected int outputHeight; // assumed equal to outputWidth
        protected int outputWidth;

        #endregion


        #region Layer base class properties (public)

        public string Type
        {
            get { return type; }
            //set { this.type = value; } // shouldn't need this. TO-DELETE
        }

        public int ID
        {
            get { return id; }
            set { this.id = value; }
        }

        public virtual Neurons Input
        {
            get { return inputNeurons; }
            set { inputNeurons = value; }
        }

        public virtual Neurons Output
        {
            get { return outputNeurons; }
        }

        public int OutputDepth // allows setup of next layer
        {
            get { return outputDepth; }
        }

        public int OutputHeight // allows setup of next layer
        {
            get { return outputHeight; }
        }

        public int OutputWidth // allows setup of next layer
        {
            get { return outputWidth; }
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


        #region Training methods

        /// <summary>
        /// Feed data into the network. Only implemented by InputLayer.
        /// </summary>
        /// <param name="dataSet"></param>
        /// <param name="iDataPoint"></param>
        public virtual void Feed(DataSet dataSet, int iDataPoint)
        {
        }

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
