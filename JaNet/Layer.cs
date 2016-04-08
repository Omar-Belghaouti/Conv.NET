using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    // CLEAN

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

        public int NInputUnits
        {
            get { return nInputUnits; }
        }

        public int NOutputUnits
        {
            get { return nOutputUnits; }
        }

        public virtual Neurons InputNeurons
        {
            get { return inputNeurons; }
            set { inputNeurons = value; }
        }

        public virtual Neurons OutputNeurons
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


        #region Setup methods

        /// <summary>
        /// Connect this layer to previous one.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public virtual void ConnectTo(Layer PreviousLayer)
        {
            this.InputNeurons = PreviousLayer.OutputNeurons; // assignment by reference! 
            // In memory, output neurons of previous layer and input neurons of current layer are the same thing!
        }


        /// <summary>
        /// Initialize layer parameters.
        /// </summary>
        public virtual void InitializeParameters()
        {
            // Base class: just make sure output neurons exist (i.e. this method is called AFTER method ConnectTo() )
            if (outputNeurons == null)
                throw new MissingFieldException("Cannot call InitializeParameters() if parameters do not exist yet! Make sure layer has been connected.");
        }

        #endregion


        #region Operating methods


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

        #endregion

        // TODO: kill this
        #region Debugging

        public virtual void DisplayParameters()
        {

        }


        #endregion

    }
}
