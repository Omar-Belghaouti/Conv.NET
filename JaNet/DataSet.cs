using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using System.Globalization;
using OpenCL.Net;

namespace JaNet
{
    [Serializable]
    class DataSet
    {
        // TODO: clean code

        #region Fields

        private int size;
        private int nClasses;
        private int dataDimension;

#if OPENCL_ENABLED
        private List<Mem> dataGPU;
        //private List<Mem> labelsGPU;
        //private List<Mem> labelArraysGPU;
#else
        private List<double[]> data;
        //private List<int[]> labelArrays;
#endif

        private List<int> labels;

        #endregion


        #region Properties

        public int Size
        {
            get { return size; }
        }

        public int NumberOfClasses
        {
            get { return nClasses; }
        }

        public int DataDimension
        {
            get { return dataDimension; }
        }

        public List<int> Labels
        {
            get { return this.labels; }
        }

#if OPENCL_ENABLED

        public List<Mem> DataGPU
        {
            get{ return this.dataGPU; }
        }

        /*
        public List<Mem> LabelsGPU
        {
            get { return this.labelsGPU; }
        }

        public Mem LabelArraysGPU(int iExample)
        {
            return labelArraysGPU[iExample];
        }
        */
#else
        public List<double[]> Data
        {
            get{ return this.data; }
        }

        /*
        public int[] GetLabelArray(int Index)
        {
            return this.labelArrays[Index];
        }
        */
#endif

        #endregion


        #region Constructor

        /// <summary>
        /// Constructor of DataSet class. Data and labels in two separate text files.
        /// </summary>
        /// <param name="nClasses"></param>
        /// <param name="dataPath"></param>
        public DataSet(int nClasses)
        {
            new System.Globalization.CultureInfo("en-US");

            this.nClasses = nClasses;
            this.size = 0;

            // Initialize empty lists

#if OPENCL_ENABLED
            this.dataGPU = new List<Mem>();
#else
            this.data = new List<double[]>();
            //this.labelArrays = new List<int[]>();
#endif
            this.labels = new List<int>();
        }

        #endregion


        #region Methods

        public void ReadData(string dataPath)
        {
            // Read images
            foreach (var line in System.IO.File.ReadAllLines(dataPath))
            {
                this.size += 1;

                var columns = line.Split('\t');
                this.dataDimension = columns.Length;

#if OPENCL_ENABLED
                float[] dataPoint = new float[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    dataPoint[i] = float.Parse(columns[i], CultureInfo.InvariantCulture.NumberFormat);
                }
                int datumBytesSize = sizeof(float) * dataPoint.Length;
                Mem tmpBuffer = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                            MemFlags.ReadOnly | MemFlags.CopyHostPtr,
                                                            (IntPtr)datumBytesSize,
                                                            dataPoint,
                                                            out OpenCLSpace.ClError);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "DataSet(): Cl.CreateBuffer tmpBuffer");
                this.dataGPU.Add(tmpBuffer);
#else
                double[] image = new double[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    image[i] = double.Parse(columns[i], CultureInfo.InvariantCulture.NumberFormat);
                }
                this.data.Add(image);
#endif
            }
        }

        public void ReadLabels(string labelsPath)
        {
            // Read labels
            foreach (var line in System.IO.File.ReadAllLines(labelsPath))
            {
                int label = Convert.ToInt16(line);
                //int[] labelArray = new int[nClasses];
                //labelArray[label] = 1;

                this.labels.Add(label);
                /*
#if OPENCL_ENABLED
                
                Mem tmpBufferLabel = (Mem)Cl.CreateBuffer(  OpenCLSpace.Context,
                                                            MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                            (IntPtr)sizeof(int),
                                                            label,
                                                            out OpenCLSpace.ClError);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "DataSet(): Cl.CreateBuffer tmpBufferLabel");
                this.labelsGPU.Add(tmpBufferLabel);
                
                Mem tmpBufferLabelArray = (Mem)Cl.CreateBuffer( OpenCLSpace.Context,
                                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                                (IntPtr)(sizeof(int) * nClasses),
                                                                labelArray,
                                                                out OpenCLSpace.ClError);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "DataSet(): Cl.CreateBuffer tmpBufferLabelArray");
                this.labelArraysGPU.Add(tmpBufferLabelArray);
#else
                this.labelArrays.Add(labelArray);
#endif
                */
            }

            Console.WriteLine("\tImported {0} images. \n\tImage dimension: {1}.\n", size, dataDimension);
        }

        #endregion
    }
}
