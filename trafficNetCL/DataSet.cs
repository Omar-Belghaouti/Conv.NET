using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;
using System.Globalization;

namespace TrafficNetCL
{
    class DataSet
    {

        private List<float[]> data;
        private List<int> labels;
        private List<float[]> labelArrays;
        private int nClasses;
        private int size;

        /*
        public void DataAdd(float[] dataPoint, int label, float[] labelArray)
        {
            this.data.Add(dataPoint);
            this.labels.Add(label);
            this.labelArrays.Add(labelArray);
            this.size += 1;
        }
         * */

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

        public int Size
        {
            get { return size; }
        }

        public int NumberOfClasses
        {
            get { return nClasses; }
        }

        /// <summary>
        ///  Constructor 1: both data AND labels in the same text file
        /// </summary>
        /// <param name="nClasses"></param>
        /// <param name="dataPath"></param>
        public DataSet(int nClasses, string dataPath)
        {
            Console.WriteLine("Importing data set from file {0}...", dataPath);

            new System.Globalization.CultureInfo("en-US");

            // Initialize empty lists
            this.data = new List<float[]>();
            this.labels = new List<int>();
            this.labelArrays = new List<float[]>();
            this.nClasses = nClasses;
            this.size = 0;

            foreach (var line in System.IO.File.ReadAllLines(dataPath))
            {
                var columns = line.Split('\t');

                //Console.WriteLine("Line {0} reads:", this.size);

                float[] dataPoint = new float[columns.Length - 1];
                for (int i = 0; i < columns.Length - 1; i++)
                {
                    dataPoint[i] = float.Parse(columns[i], CultureInfo.InvariantCulture.NumberFormat);
                    //Console.Write("{0}  ", columns[i].Trim());
                }

                int label = Convert.ToInt16(columns[columns.Length - 1]);
                if (label == -1)
                    label = 0;

                float[] labelArray = new float[nClasses];
                labelArray[label] = 1.0f;

                this.data.Add(dataPoint);
                this.labels.Add(label);
                this.labelArrays.Add(labelArray);
                this.size += 1;
            }


            Console.WriteLine("\tImported {0} data points. \n\tData is {1} dimensional.\n", this.size, this.GetDataPoint(0).Length);
        }

        /// <summary>
        /// Constructor 2: data and labels in two separate text files (paths as arguments)
        /// </summary>
        /// <param name="nClasses"></param>
        /// <param name="dataPath"></param>
        public DataSet(int nClasses, string imagesPath, string labelsPath)
        {
            new System.Globalization.CultureInfo("en-US");

            // Initialize empty lists
            this.data = new List<float[]>();
            this.labels = new List<int>();
            this.labelArrays = new List<float[]>();
            this.nClasses = nClasses;
            this.size = 0;

            // Read images
            foreach (var line in System.IO.File.ReadAllLines(imagesPath))
            {
                var columns = line.Split('\t');

                float[] image = new float[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    image[i] = float.Parse(columns[i], CultureInfo.InvariantCulture.NumberFormat);
                }

                this.data.Add(image);
            }

            // Read labels
            foreach (var line in System.IO.File.ReadAllLines(labelsPath))
            {
                int label = Convert.ToInt16(line);
                float[] labelArray = new float[nClasses];
                labelArray[label] = 1.0f;

                this.labels.Add(label);
                this.labelArrays.Add(labelArray);

                this.size += 1;
            }

            Console.WriteLine("\tImported {0} images. \n\tImage dimension: {1}.\n", this.size, this.GetDataPoint(0).Length);
        }


    }
}
