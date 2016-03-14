using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    class Tanh : Layer
    {

        private int numberOfUnits;
        private double beta;
        

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
            //this.input = new Neurons(this.numberOfUnits);
            this.output = new Neurons(this.numberOfUnits);
            this.deltaInput = new float[this.numberOfUnits];
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
                tmpOutput[i] = (float)Math.Tanh(this.input.Get()[i]);
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
            if (this.DeltaInput.Length != nextLayer.DeltaInput.Length)
                throw new System.InvalidOperationException("Tanh layer: mismatch in length of delta arrays.");
            
            for (int i = 0; i < this.numberOfUnits; i++)
                this.DeltaInput[i] = this.nextLayer.DeltaInput[i] * (1 - (float)Math.Pow((double)this.output.Get()[i], 2));

        }

        public override void BackPropOneCPU(float[] CostGradient) // overloaded in case this is last layer, takes gradient of cost as argument
        {
            if (this.DeltaInput.Length != CostGradient.Length)
                throw new System.InvalidOperationException("Output layer: mismatch between length of delta array and gradient of cost function.");

            for (int i = 0; i < this.numberOfUnits; i++)
                this.DeltaInput[i] = CostGradient[i] * (1 - (float)Math.Pow((double)this.output.Get()[i], 2));

        }

        public override void BackPropBatchCPU()
        {
            // TO-DO
        }

        public override void UpdateParameters(double learningRate)
        {
            // nothing to update
        }

        #endregion


    }
}
