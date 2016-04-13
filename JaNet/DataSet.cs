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
    class DataSet
    {
        // CLEAN

        #region DataSet class fields

        private int nPointsPerImage;
        private int size;
        private int nClasses;


#if OPENCL_ENABLED
        private List<Mem> dataGPU;
        private List<Mem> labelsGPU;
        private List<Mem> labelArraysGPU;
#else
        private List<double[]> data;
        private List<int[]> labelArrays;
#endif

        private List<int> labels;

        #endregion


        #region DataSet class properties

        public int Size
        {
            get { return size; }
        }

        public int NumberOfClasses
        {
            get { return nClasses; }
        }

        public int GetLabel(int Index)
        {
            return this.labels[Index];
        }

#if OPENCL_ENABLED
        // Pseudo-indexers for GPU buffers

        public Mem DataGPU(int iExample)
        {
            return dataGPU[iExample];
        }

        public Mem LabelsGPU(int iExample)
        {
            return labelsGPU[iExample];
        }

        public Mem LabelArraysGPU(int iExample)
        {
            return labelArraysGPU[iExample];
        }
#else
        public double[] GetDataPoint(int Index)
        {
            return this.data[Index];
        }

        public int[] GetLabelArray(int Index)
        {
            return this.labelArrays[Index];
        }
#endif

        #endregion

        /// <summary>
        /// Constructor of DataSet class. Data and labels in two separate text files.
        /// </summary>
        /// <param name="nClasses"></param>
        /// <param name="dataPath"></param>
        public DataSet(int nClasses, string imagesPath, string labelsPath)
        {
            new System.Globalization.CultureInfo("en-US");

            this.nClasses = nClasses;
            this.size = 0;

            // Initialize empty lists

#if OPENCL_ENABLED
            this.dataGPU = new List<Mem>();
            this.labelsGPU = new List<Mem>();
            this.labelArraysGPU = new List<Mem>();
#else
            this.data = new List<double[]>();
            this.labelArrays = new List<int[]>();
#endif
            this.labels = new List<int>();

            // Read images
            foreach (var line in System.IO.File.ReadAllLines(imagesPath))
            {
                this.size += 1;

                var columns = line.Split('\t');
                this.nPointsPerImage = columns.Length;

#if OPENCL_ENABLED
                float[] image = new float[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    image[i] = float.Parse(columns[i], CultureInfo.InvariantCulture.NumberFormat);
                }
                int imageBytesSize = sizeof(float) * image.Length;
                Mem tmpBufferImage = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                            (IntPtr)imageBytesSize,
                                                            image,
                                                            out OpenCLSpace.ClError);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "DataSet(): Cl.CreateBuffer tmpBufferImage");
                this.dataGPU.Add(tmpBufferImage);
#else
                double[] image = new double[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    image[i] = double.Parse(columns[i], CultureInfo.InvariantCulture.NumberFormat);
                }
                this.data.Add(image);
#endif


#if OPENCL_ENABLED
                
#endif

            }

            // Read labels
            foreach (var line in System.IO.File.ReadAllLines(labelsPath))
            {
                int label = Convert.ToInt16(line);
                int[] labelArray = new int[nClasses];
                labelArray[label] = 1;

                this.labels.Add(label);
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
            }

            Console.WriteLine("\tImported {0} images. \n\tImage dimension: {1}.\n", size, nPointsPerImage);
        }

    }
}
