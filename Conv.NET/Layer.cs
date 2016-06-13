using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace Conv.NET
{
    // CLEAN

    /// <summary>
    /// Base layer class. All other layers will inherit from this.
    /// </summary>
    [Serializable]
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
        }

        public int ID
        {
            get { return id; }
            set { this.id = value; }
        }

        public int NInputUnits
        {
            get { return nInputUnits; }
            set { this.nInputUnits = value; }
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

        public int InputDepth // allows setup of next layer
        {
            get { return inputDepth; }
            set { inputDepth = value; }
        }

        public int InputHeight // allows setup of next layer
        {
            get { return inputHeight; }
            set { inputHeight = value; }
        }

        public int InputWidth // allows setup of next layer
        {
            get { return inputWidth; }
            set { inputWidth = value; }
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

        // Some virtual properties

        public virtual double DropoutParameter
        {
            set { }
        }

        public virtual bool IsEpochBeginning
        {
            set { }
        }
        public virtual bool IsTraining
        {
            set { }
        }
        public virtual bool IsPreInference
        {
            set { }
        }
        public virtual bool IsInference
        {
            set { }
        }

        public virtual Mem WeightsGPU
        {
            get { return new Mem();  }
        }

        public virtual int FilterSize
        {
            get { return 0; }
        }

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
        public virtual void InitializeParameters(string Option)
        {
            // Base class: just make sure output neurons exist (i.e. this method is called AFTER method ConnectTo() )
            if (outputNeurons == null)
                throw new MissingFieldException("Cannot call InitializeParameters(). Layer has not been connected yet.");
        }

        /// <summary>
        ///  Setup global/local work group sizes used by OpenCL kernels.
        /// </summary>
        public virtual void SetWorkGroups()
        {
            // Implementation is layer type-specific.
        }


        public virtual void CopyBuffersToHost()
        {

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
        public virtual void UpdateSpeeds(double learningRate, double momentumMultiplier, double weightDecayCoefficient)
        {
        }

        /// <summary>
        /// Updates layer parameters (if any)
        /// </summary>
        public virtual void UpdateParameters(double weightMaxNorm)
        {
        }

        #endregion


        #region Gradient check

        public virtual double[] GetParameters()
        {
            return null;
        }

        public virtual double[] GetParameterGradients()
        {
            return null;
        }

        public virtual void SetParameters(double[] NewParameters)
        {
        }

        public virtual double[] GetInput()
        {
            int inputArraySize = nInputUnits * inputNeurons.MiniBatchSize;
            double[] input = new double[inputArraySize];

            // Copy device buffer to host
            float[] tmpInput = new float[inputArraySize];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        inputNeurons.ActivationsGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputArraySize),
                                                        tmpInput,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            // Convert to double and write into public fields
            for (int i = 0; i < inputArraySize; ++i)
            {
                input[i] = (double)tmpInput[i];
            }

            return input;
        }

        public virtual double[] GetInputGradients()
        {
            int inputArraySize = nInputUnits * inputNeurons.MiniBatchSize;
            double[] inputGradients = new double[inputArraySize];

            // Copy device buffer to host
            float[] tmpInputGradients = new float[inputArraySize];
            OpenCLSpace.ClError = Cl.EnqueueReadBuffer(OpenCLSpace.Queue,
                                                        inputNeurons.DeltaGPU, // source
                                                        Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputArraySize),
                                                        tmpInputGradients,  // destination
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "clEnqueueReadBuffer");

            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");

            // Convert to double and write into public fields
            for (int i = 0; i < inputArraySize; ++i)
            {
                inputGradients[i] = (double)tmpInputGradients[i];
            }

            return inputGradients;
        }

        public virtual void SetInput(double[] NewInput)
        {
            // Convert to float and write into tmp arrays
            int inputArraySize = nInputUnits * inputNeurons.MiniBatchSize;
            float[] tmpInput = new float[inputArraySize];
            for (int i = 0; i < inputArraySize; ++i)
                tmpInput[i] = (float)NewInput[i];

            // Write arrays into buffers on device

            OpenCLSpace.ClError = Cl.EnqueueWriteBuffer(OpenCLSpace.Queue,
                                                        inputNeurons.ActivationsGPU,
                                                        OpenCL.Net.Bool.True,
                                                        (IntPtr)0,
                                                        (IntPtr)(sizeof(float) * inputArraySize),
                                                        tmpInput,
                                                        0,
                                                        null,
                                                        out OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.EnqueueWriteBuffer");
            OpenCLSpace.ClError = Cl.ReleaseEvent(OpenCLSpace.ClEvent);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.ReleaseEvent");

            OpenCLSpace.ClError = Cl.Finish(OpenCLSpace.Queue);
            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Cl.Finish");
        }

        #endregion
    }
}
