using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    class ReLU : Layer
    {
        #region ReLU layer class fields (private)

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
        public ReLU()
        {
            Console.WriteLine("Adding a ReLU layer...");
            this.layerType = "ReLU";
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
            throw new System.InvalidOperationException("You are setting a ReLU layer as first layer of the network...\nIs it really what you want to do?");
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
                if (this.input.Get()[i] > 0)
                    tmpOutput[i] = this.input.Get()[i];
                else
                    tmpOutput[i] = 0.0F;
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
            for (int i = 0; i < this.numberOfUnits; i++)
                this.input.Delta[i] = this.input.Get()[i] > 0 ? this.output.Delta[i] : 0.0F;
        }

        public override void BackPropBatchCPU()
        {
            float[] tmpDelta = new float[numberOfUnits];
            for (int i = 0; i < this.numberOfUnits; i++)
                this.Input.Delta[i] = this.Input.Get()[i] > 0 ? this.Output.Delta[i] : 0.0F;

            this.Input.Delta = this.Input.Delta.Zip(tmpDelta, (x, y) => x + y).ToArray();
        }

        public override void UpdateParameters(double learningRate, double momentumCoefficient)
        {
            // nothing to update
        }

        #endregion


    }
}
