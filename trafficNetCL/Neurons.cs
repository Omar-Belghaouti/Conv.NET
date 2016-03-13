using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    public class Neurons
    {
        private int nUnits;
        private float[] unitsActivations;

        public int NumberOfUnits 
        {
            get { return nUnits;  }
        }

        public Neurons() // Generic constructor
        {
        }

        public Neurons(int NumberOfUnits) // Constructor for 1D Neurons instance
        {
            this.nUnits = NumberOfUnits;
            this.unitsActivations = new float[nUnits];
        }

        /*
        public Neurons(int width, int height, int depth) // Constructor for 3D Neurons instance
        {
            this.nUnits = width * height * depth;
            this.unitsActivations = new float[width, height, depth];
            this.typeOfUnits = typeof(float[, ,]);
        }
         * */

        public void Set(float[] value)
        {
            this.unitsActivations = value;
        }

        public float[] Get()
        {
            return unitsActivations;
        }


        /*
        public void Set(float[,,] value)
        {
            // TO-DO: improve error-handling
            if (typeOfUnits.Equals(typeof(float[, ,]))) // simple case (same type of neurons)
                this.unitsActivations = value;
            else if (typeOfUnits.Equals(typeof(float[]))) // multi-dim neuron array to one-dim
                this.unitsActivations = value.Cast<float>().ToArray();
            else
                Console.WriteLine("Something wrong happened when setting neuron values!");

        }
         * */

        
        /*
        public void ConnectTo(Neurons PreviousOutput)
        {
            this.unitsActivations = new object();
            this.unitsActivations = PreviousOutput.Get();
            this.nUnits = PreviousOutput.NumberOfUnits;
            this.typeOfUnits = unitsActivations.GetType();
        }
         * */
        
        /*
        public void Setup(int nUnits) // if 1-dim array of units
        {
            unitsActivations = new float[nUnits];
            typeOfUnits = typeof(float[]);
        }

        public void Setup(int width, int height, int depth) // if 3-dim array of units
        {
            unitsActivations = new float[width, height, depth];
            typeOfUnits = typeof(float[,,]);
        }
         * */
        
        
    }
}
