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

        /// <summary>
        /// Constructor of Tanh layer. Specify beta parameter as argument.
        /// </summary>
        /// <param name="Beta"></param>
        public Tanh(double Beta)
        {
            Console.WriteLine("Adding a tanh layer with activation parameter {0}...", Beta);
            this.beta = Beta;
        }

        /// <summary>
        ///  Connect current layer to layer given as argument.
        /// </summary>
        /// <param name="PreviousLayer"></param>
        public override void ConnectTo(Layer PreviousLayer)
        {
            this.numberOfUnits = PreviousLayer.Output.NumberOfUnits;
            this.input = new Neurons(this.numberOfUnits);
            this.input.ConnectTo(PreviousLayer.Output);
            this.output = new Neurons(this.numberOfUnits);
        }

        /// <summary>
        /// Method to set this layer as the first layer of the network.
        /// </summary>
        public override void SetAsFirstLayer(int InputImageWidth, int InputImageHeight, int InputImageDepth)
        {
            throw new System.InvalidOperationException("You are setting a sigmoid layer as first layer of the network...\nIs it really what you want to do?");
        }

        public override void InitializeParameters()
        {
            // This layer doesn't learn: No parameters to initialize.
        }


        public override void ForwardOneCPU()
        {
            // TO-DO (all)
            /*
            float[] tmpOutput = new float[numberOfUnits];
            for (int i = 0; i < numberOfUnits; i++)
            {
                tmpOutput[i] = 
            }
            this.output.Set(this.input.Get().ToArray().Select(x => Math.Tanh(x)).ToArray());
             * */
        }

        public override void BackPropOneCPU()
        {
            
        }
    }
}
