using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    class Tanh : Layer
    {
        #region Tanh layer class fields (private)

        private double beta;

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
        public Tanh(double Beta)
        {
            Console.WriteLine("Adding a tanh layer with activation parameter {0}...", Beta);

            this.beta = Beta;
            this.layerType = "Tanh";
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
            throw new System.InvalidOperationException("You are setting a sigmoid layer as first layer of the network...\nIs it really what you want to do?");
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
                tmpOutput[i] = (float)Math.Tanh(beta*this.input.Get()[i]);
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
                this.input.Delta[i] = this.output.Delta[i] * (float) (beta * (1 - Math.Pow((double)this.output.Get()[i], 2)) );

        }

        public override void BackPropBatchCPU()
        {
            // TO-DO

            float[] tmpDelta = new float[numberOfUnits];
            for (int i = 0; i < this.numberOfUnits; i++)
                tmpDelta[i] = this.output.Delta[i] * (float)(beta * (1 - Math.Pow((double)this.output.Get()[i], 2)));
            
            this.input.Delta = this.input.Delta.Zip(tmpDelta, (x, y) => x + y).ToArray();

        }

        public override void UpdateParameters(double learningRate, double momentumCoefficient)
        {
            // nothing to update
        }

        #endregion


    }
}
