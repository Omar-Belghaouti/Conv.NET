using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
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
            Console.WriteLine("Adding a SoftMax layer...");

            this.layerType = "SoftMax";
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
        public override void SetAsFirstLayer(int[] InputDimensions)
        {
            throw new System.InvalidOperationException("You are setting a SoftMax layer as first layer of the network...\nIs it really what you want to do?");
        }

        public override void InitializeParameters()
        {
            // This layer doesn't learn: No parameters to initialize.
        }

        #endregion


        #region Training methods

        public override void ForwardOneCPU()
        {
            float[] tmpOutput = new float[this.numberOfUnits];
            for (int i = 0; i < this.numberOfUnits; i++)
            {
                tmpOutput[i] = (float)Math.Exp(this.input.Get()[i]);
            }
            float sum = tmpOutput.Sum();
            for (int i = 0; i < this.numberOfUnits; i++)
            {
                tmpOutput[i] /= sum;
            }

            this.output.Set(tmpOutput);
        }

        public override void ForwardBatchCPU()
        {
        }

        public override void ForwardGPU()
        {
        }

        public override void BackPropOneCPU()
        {
            if (this.Input.Delta.Length != this.Output.Delta.Length)
                throw new System.InvalidOperationException("Softmax layer: mismatch in length of delta arrays.");

            // TO-DO: implement this backprop
        }

        public override void BackPropBatchCPU()
        {
            // TO-DO
        }

        public override void UpdateParameters(double learningRate, double momentumCoefficient)
        {
            // nothing to update
        }

        #endregion


    }
}
