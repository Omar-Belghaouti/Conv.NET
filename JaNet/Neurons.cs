using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OpenCL.Net;

namespace JaNet
{
    public class Neurons
    {
        #region Neuron class fields (private)

        private int nUnits;

#if OPENCL_ENABLED
        private Mem activationsGPU;
        private Mem deltaGPU;
#else
        private float[] activations;
        private float[] delta;
#endif

        #endregion

        #region Neuron class properties (public)

        public int NumberOfUnits
        {
            get { return nUnits; }
        }

#if OPENCL_ENABLED

        public Mem ActivationsGPU 
        { 
            get { return this.activationsGPU; }
            set { this.activationsGPU = value; }
        }

        public Mem DeltaGPU 
        { 
            get { return this.deltaGPU; }
            set { this.deltaGPU = value; }
        }
#else       
        public float[] GetHost()
        {
            return activations;
        }

        public void SetHost(float[] value)
        {
            this.activations = value;
        }

        public float[] DeltaHost
        {
            get { return delta; }
            set { this.delta = value; }
        }

#endif

        #endregion


        #region Constructors

        public Neurons() // Generic constructor
        {
        }

        /// <summary>
        /// Constructor. Use for CPU.
        /// </summary>
        /// <param name="NumberOfUnits"></param>
        public Neurons(int NumberOfUnits)
        {
            this.nUnits = NumberOfUnits;

#if OPENCL_ENABLED
            int bufferSize = sizeof(float) * NumberOfUnits;
            
            this.activationsGPU = (Mem)Cl.CreateBuffer(CL.Context, MemFlags.ReadWrite, (IntPtr)bufferSize, out CL.Error);
            this.deltaGPU = (Mem)Cl.CreateBuffer(CL.Context, MemFlags.ReadWrite, (IntPtr)bufferSize, out CL.Error);
            CL.CheckErr(CL.Error, "Neurons constructor: Cl.CreateBuffer");
#else

            this.activations = new float[nUnits];
            this.delta = new float[nUnits];
#endif
        
        }
        
        #endregion



    }
}
