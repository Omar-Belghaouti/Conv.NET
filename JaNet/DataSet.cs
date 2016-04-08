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
    public class DataSet
    {
        #region DataSet class fields

        private int size;
        private int nClasses;

        private List<float[]> data;
        private List<int> labels;
        private List<float[]> labelArrays;

#if OPENCL_ENABLED
        private List<Mem> dataGPU;
        private List<Mem> labelsGPU;
        private List<Mem> labelArraysGPU;
#endif

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

        public float[] GetDataPoint(int Index)
        {
            return this.data[Index];
        }

        public int GetLabel(int Index)
        {
            return this.labels[Index];
        }

        public float[] GetLabelArray(int Index)
        {
            return this.labelArrays[Index];
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
#endif

        #endregion


        #region Constructor

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
            this.data = new List<float[]>();
            this.labels = new List<int>();
            this.labelArrays = new List<float[]>();
            
#if OPENCL_ENABLED
            ErrorCode clError = new ErrorCode();

            this.dataGPU = new List<Mem>();
            this.labelsGPU = new List<Mem>();
            this.labelArraysGPU = new List<Mem>();
#endif

            // Read images
            foreach (var line in System.IO.File.ReadAllLines(imagesPath))
            {
                this.size += 1;

                var columns = line.Split('\t');

                float[] image = new float[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    image[i] = float.Parse(columns[i], CultureInfo.InvariantCulture.NumberFormat);
                }

                this.data.Add(image);

#if OPENCL_ENABLED
                int imageBytesSize = sizeof(float) * image.Length;
                Mem tmpBufferImage = (Mem) Cl.CreateBuffer( OpenCLSpace.Context, 
                                                            MemFlags.ReadWrite | MemFlags.CopyHostPtr, 
                                                            (IntPtr)imageBytesSize, 
                                                            image, 
                                                            out clError);
                OpenCLSpace.CheckErr(clError, "DataSet(): Cl.CreateBuffer tmpBufferImage");
                this.dataGPU.Add(tmpBufferImage);
#endif

            }

            // Read labels
            foreach (var line in System.IO.File.ReadAllLines(labelsPath))
            {
                int label = Convert.ToInt16(line);
                float[] labelArray = new float[nClasses];
                labelArray[label] = 1.0f;

                this.labels.Add(label);
                this.labelArrays.Add(labelArray);

#if OPENCL_ENABLED
                Mem tmpBufferLabel = (Mem) Cl.CreateBuffer( OpenCLSpace.Context, 
                                                            MemFlags.ReadWrite | MemFlags.CopyHostPtr,
                                                            (IntPtr)sizeof(int), 
                                                            label, 
                                                            out clError);
                OpenCLSpace.CheckErr(clError, "DataSet(): Cl.CreateBuffer tmpBufferLabel");
                this.labelsGPU.Add(tmpBufferLabel);

                Mem tmpBufferLabelArray = (Mem)Cl.CreateBuffer( OpenCLSpace.Context, 
                                                                MemFlags.ReadWrite | MemFlags.CopyHostPtr, 
                                                                (IntPtr)(sizeof(int) * nClasses),
                                                                labelArray, 
                                                                out clError);
                OpenCLSpace.CheckErr(clError, "DataSet(): Cl.CreateBuffer tmpBufferLabelArray");
                this.labelArraysGPU.Add(tmpBufferLabelArray);
#endif                
            }

            Console.WriteLine("\tImported {0} images. \n\tImage dimension: {1}.\n", this.size, this.GetDataPoint(0).Length);
        }

        #endregion
    }
}
