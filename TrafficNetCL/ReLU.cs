using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JaNet
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
            //Console.WriteLine("Adding a ReLU layer...");
            this.type = "ReLU";
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
            throw new System.InvalidOperationException("You are setting a ReLU layer as first layer of the network...\nIs it really what you want to do?");
        }

        public override void InitializeParameters()
        {
            // This layer doesn't learn: No parameters to initialize.
        }

        #endregion


        #region Training methods

        public override void FeedForward()
        {
            float[] tmpOutput = new float[this.numberOfUnits];
            for (int i = 0; i < this.numberOfUnits; i++)
            {
                if (this.input.GetHost()[i] > 0)
                    tmpOutput[i] = this.input.GetHost()[i];
                else
                    tmpOutput[i] = 0.0F;
            }
            this.output.SetHost(tmpOutput);
        }

        public override void BackPropagate()
        {
            for (int i = 0; i < this.numberOfUnits; i++)
                this.input.DeltaHost[i] = this.input.GetHost()[i] > 0 ? this.output.DeltaHost[i] : 0.0F;
        }

        /*
        public override void BackPropBatchCPU()
        {
            float[] tmpDelta = new float[numberOfUnits];
            for (int i = 0; i < this.numberOfUnits; i++)
                this.Input.DeltaHost[i] = this.Input.GetHost()[i] > 0 ? this.Output.DeltaHost[i] : 0.0F;

            this.Input.DeltaHost = this.Input.DeltaHost.Zip(tmpDelta, (x, y) => x + y).ToArray();
        }
        */
        

        #endregion


    }
}
