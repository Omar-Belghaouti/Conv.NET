using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
        /// Link this layer to previous one.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public virtual void ConnectTo(Layer PreviousLayer)
        {
            this.Input = PreviousLayer.Output; // assignment by reference! 
            // In memory, output neurons of previous layer and input neurons of current layer are the same thing!
        }


        /// <summary>
        /// Set this as the first layer of the network.
        /// </summary>
        /// 
        public virtual void SetAsFirstLayer(int InputWidth, int InputHeight, int InputDepth)
        {

        }

        /// <summary>
        /// Initialize layer parameters.
        /// </summary>
        public abstract void InitializeParameters();

        #endregion


        #region Training methods (public abstract)

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
        public abstract void UpdateParameters(double learningRate, double momentumMultiplier);

        public virtual void ClearDelta()
        {
            Array.Clear(this.input.Delta, 0, this.input.NumberOfUnits);
            Array.Clear(this.output.Delta, 0, this.output.NumberOfUnits);
        }

        

        #endregion

        // DEBUGGING METHODS
        public virtual void DisplayParameters()
        {

        }

        public virtual void DisplayDeltas()
        {


            Console.WriteLine("Layer INPUT deltas:");
            for (int i = 0; i < this.input.NumberOfUnits; i++)
            {
                Console.Write("{0}  ", this.input.Delta[i]);
            }

            Console.WriteLine("\nLayer OUTPUT deltas:");
            for (int i = 0; i < this.output.NumberOfUnits; i++)
            {
                Console.Write("{0}  ", this.output.Delta[i]);
            }

        }

    }
}
