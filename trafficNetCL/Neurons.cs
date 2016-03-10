using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    public class Neurons
    {

        private object neuronActivations;
        Type typeOfNeurons;

        public Neurons() // Generic constructor
        {
        }

        public Neurons(int nUnits) // Constructor for 1D Neurons instance
        {
            neuronActivations = new float[nUnits];
            typeOfNeurons = typeof(float[]);
        }

        public Neurons(int width, int height, int depth) // Constructor for 3D Neurons instance
        {
            neuronActivations = new float[width, height, depth];
            typeOfNeurons = typeof(float[, ,]);
        }

        public void Set(float[] value)
        {
            // TO-DO: improve error-handling
            if (typeOfNeurons.Equals(typeof(float[]))) // simple case (same type of neurons)
                this.neuronActivations = value;
            else
                Console.WriteLine("Something wrong happened when setting neuron values!");
        }

        public void Set(float[,,] value)
        {
            // TO-DO: improve error-handling
            if (typeOfNeurons.Equals(typeof(float[, ,]))) // simple case (same type of neurons)
                this.neuronActivations = value;
            else if (typeOfNeurons.Equals(typeof(float[]))) // multi-dim neuron array to one-dim
                this.neuronActivations = value.Cast<float>().ToArray();
            else
                Console.WriteLine("Something wrong happened when setting neuron values!");

        }

        public object Get()
        {
            return neuronActivations;
        }

        public void ConnectTo(Neurons PreviousOutput)
        {
            Console.WriteLine("Output neurons of previous layer are of type {0}", PreviousOutput.Get().GetType());

            this.neuronActivations = new object();
            neuronActivations = PreviousOutput.Get();
            typeOfNeurons = neuronActivations.GetType();

            Console.WriteLine("Setting input neurons of this layer as {0}", typeOfNeurons);
        }

        public void Setup(int nUnits) // if 1-dim array of units
        {
            neuronActivations = new float[nUnits];
            typeOfNeurons = typeof(float[]);
        }

        public void Setup(int width, int height, int depth) // if 3-dim array of units
        {
            neuronActivations = new float[width, height, depth];
            typeOfNeurons = typeof(float[,,]);
        }
        
        
    }
}
