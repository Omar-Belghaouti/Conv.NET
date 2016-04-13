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
        // CLEAN 

        #region Neuron class fields (private)

        private int nUnits;
        private int miniBatchSize = 1;

#if OPENCL_ENABLED
        private List<Mem> activationsGPU;
        private List<Mem> deltaGPU;
#else
        private List<double[]> activations;
        private List<double[]> delta;
#endif

        #endregion


        #region Neuron class properties (public)

        public int NumberOfUnits
        {
            get { return nUnits; }
        }

        public int MiniBatchSize
        {
            get { return miniBatchSize; }
            set { this.miniBatchSize = value; }
        }

#if OPENCL_ENABLED

        public List<Mem> ActivationsGPU 
        { 
            get { return this.activationsGPU; }
            set { this.activationsGPU = value; }
        }

        public List<Mem> DeltaGPU 
        { 
            get { return this.deltaGPU; }
            set { this.deltaGPU = value; }
        }
#else       
        public List<double[]> GetHost()
        {
            return activations;
        }

        public void SetHost(int iExample, double[] value)
        {
            this.activations[iExample] = value;
        }

        public List<double[]> DeltaHost
        {
            get { return delta; }
            set { this.delta = value; }
        }

#endif

        #endregion


        #region Constructors

        /// <summary>
        /// Neurons constructor.
        /// </summary>
        /// <param name="NumberOfUnits"></param>
        public Neurons(int NumberOfUnits)
        {
            this.nUnits = NumberOfUnits;

#if OPENCL_ENABLED
            this.activationsGPU = new List<Mem>();
            this.activationsGPU.Add( (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                            MemFlags.ReadWrite,
                                                            (IntPtr)(sizeof(float) * NumberOfUnits),
                                                            out OpenCLSpace.ClError) );

            this.deltaGPU = new List<Mem>();
            this.deltaGPU.Add( (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                    MemFlags.ReadWrite,
                                                    (IntPtr)(sizeof(float) * NumberOfUnits),
                                                    out OpenCLSpace.ClError) );

            OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Neurons constructor: Cl.CreateBuffer");
#else
            this.activations = new List<double[]>();
            this.activations.Add(new double[nUnits]);

            this.delta = new List<double[]>();
            this.delta.Add(new double[nUnits]);
#endif

        }
        
        #endregion

        public void SetupMiniBatch(int MiniBatchSize)
        {
            this.miniBatchSize = MiniBatchSize;

            for (int m = 1; m < MiniBatchSize; m++) // add MiniBatchSize - 1 buffers (to the existing list of one element)
            {
#if OPENCL_ENABLED
                this.activationsGPU.Add((Mem)Cl.CreateBuffer(   OpenCLSpace.Context,
                                                                MemFlags.ReadWrite,
                                                                (IntPtr)(sizeof(float) * NumberOfUnits),
                                                                out OpenCLSpace.ClError));
                this.deltaGPU.Add((Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                        MemFlags.ReadWrite,
                                                        (IntPtr)(sizeof(float) * NumberOfUnits),
                                                        out OpenCLSpace.ClError));
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "Neurons constructor: Cl.CreateBuffer");
#else
                this.activations.Add(new double[nUnits]);
                this.delta.Add(new double[nUnits]);
#endif
            }

        }



    }
}
