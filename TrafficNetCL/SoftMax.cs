using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JaNet
{
    class SoftMax : Layer
    {
        #region SoftMax layer class fields (private)


        /* Additional fields, inherited from "Layer" class:
        * 
        * protected Neurons input;
        * protected Neurons output;
        * 
        * protected Layer nextLayer;
        * protected string layerType;
        */

        #endregion


        #region Setup methods (to be called once)

        /// <summary>
        /// Constructor of Tanh layer. Specify beta parameter as argument.
        /// </summary>
        /// <param name="Beta"></param>
        public SoftMax()
        {
            //Console.WriteLine("Adding a SoftMax layer...");

            this.type = "SoftMax";
        }

        /// <summary>
        ///  Connect current layer to layer given as argument.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public override void ConnectTo(Layer PreviousLayer)
        {
            base.ConnectTo(PreviousLayer);

            this.numberOfUnits = PreviousLayer.Output.NumberOfUnits;
            this.output = new Neurons(this.numberOfUnits);

        }

        /// <summary>
        /// Method to set this layer as the first layer of the network.
        /// </summary>
        public override void SetAsFirstLayer(int InputWidth, int InputHeight, int InputDepth)
        {
            throw new System.InvalidOperationException("You are setting a SoftMax layer as first layer of the network...\nIs it really what you want to do?");
        }

        public override void InitializeParameters()
        {
            // This layer doesn't learn: No parameters to initialize.
        }

        #endregion


        #region Training methods

        public override void FeedForward()
        {
            // use rescaling trick to improve numerical stability
            float maxInput = this.input.GetHost()[0];
            for (int i = 1; i < this.numberOfUnits; i++)
            {
                if (this.input.GetHost()[i] > maxInput)
                    maxInput = this.input.GetHost()[i];
            }

            float[] tmpOutput = new float[this.numberOfUnits];
            for (int i = 0; i < this.numberOfUnits; i++)
            {
                tmpOutput[i] = (float)Math.Exp(this.input.GetHost()[i]-maxInput);
            }
            float sum = tmpOutput.Sum();
            for (int i = 0; i < this.numberOfUnits; i++)
            {
                tmpOutput[i] /= sum;
            }

            this.output.SetHost(tmpOutput);
        }


        public override void BackPropagate()
        {
            if (this.Input.DeltaHost.Length != this.Output.DeltaHost.Length)
                throw new System.InvalidOperationException("Softmax layer: mismatch in length of delta arrays.");

            // NO backprop here!!
            // Compute directly input.Delta from cross-entropy cost: faster and numerically more stable
        }

        #endregion


    }
}
