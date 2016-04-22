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


        #region Properties

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

#if GRADIENT_CHECK
        // accessors for gradient check 

        public virtual double[,] Weights
        {
            get { return null; }
            set { }
        }

        public virtual double[] Biases
        {
            get { return null; }
            set { }
        }

        public virtual double[,] WeightsGradients
        {
            get { return null; }
        }

        public virtual double[] BiasesGradients
        {
            get { return null; }
        }
#endif

        #endregion


        #region Setup methods

        /// <summary>
        /// Points input of this layer to output of previous layer.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public virtual void ConnectTo(Layer PreviousLayer)
        {
            this.InputNeurons = PreviousLayer.OutputNeurons; // remember that this assignment is by reference! 
            this.nInputUnits = PreviousLayer.NOutputUnits;
            
            this.inputWidth = PreviousLayer.OutputWidth;
            this.inputHeight = PreviousLayer.OutputHeight;
            this.inputDepth = PreviousLayer.OutputDepth;
        }

        /// <summary>
        /// (abstract method) Call after ConnectTo() in order to setup output units. Implementation is layer type-specific.
        /// </summary>
        public abstract void SetupOutput(); // depends on layer type

        /// <summary>
        /// Initialize layer parameters. Layer type-specific.
        /// </summary>
        public virtual void InitializeParameters()
        {
            // Base class: just make sure output neurons exist (i.e. this method is called AFTER method ConnectTo() )
            if (outputNeurons == null)
                throw new MissingFieldException("Cannot call InitializeParameters() if parameters do not exist yet! Make sure layer has been connected.");
        }

        /// <summary>
        ///  Setup global/local work group sizes used by OpenCL kernels.
        /// </summary>
        public virtual void SetWorkGroups()
        {
            // Implementation is layer type-specific.
        }

        #endregion


        #region Methods


        /// <summary>
        /// Run layer forward.
        /// </summary>
        public abstract void FeedForward();


        /// <summary>
        /// Compute errors with backpropagation.
        /// </summary>
        public abstract void BackPropagate();

        /// <summary>
        /// Updates speed of layer's parameters change (if any)
        /// </summary>
        public virtual void UpdateSpeeds(double learningRate, double momentumMultiplier)
        {
        }

        /// <summary>
        /// Updates layer parameters (if any)
        /// </summary>
        public virtual void UpdateParameters(double weightDecayCoeff)
        {
        }

        #endregion

    }
}
