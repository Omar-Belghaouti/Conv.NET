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
        private int size;

        public void Add(float[] DataPoint, int Class)
        {
            this.data.Add(DataPoint);
            this.labels.Add(Class);
            this.size += 1;
        }

        public float[] GetDataPoint(int Index)
        {
            return this.data[Index];
        }

        public int GetLabel(int Index)
        {
            return this.labels[Index];
        }

        public int Size
        {
            get { return size; }
        }

        public DataSet(string dataPath)
        {
            Console.WriteLine("Importing data set from file {0}...", dataPath);

            new System.Globalization.CultureInfo("en-US");

            // Initialize empty lists
            this.data = new List<float[]>();
            this.labels = new List<int>();
            this.size = 0;

            foreach (var line in System.IO.File.ReadAllLines(dataPath))
            {
                var columns = line.Split('\t');

                //Console.WriteLine("Line {0} reads:", this.size);

                float[] DataPoint = new float[columns.Length - 1];
                for (int i = 0; i < columns.Length - 1; i++)
                {
                    DataPoint[i] = float.Parse(columns[i], CultureInfo.InvariantCulture.NumberFormat);
                    //Console.Write("{0}  ", columns[i].Trim());
                }
                //Console.WriteLine(columns[columns.Length - 1]);
                this.Add(DataPoint, Convert.ToInt16(columns[columns.Length - 1]));
            }


            Console.WriteLine("\tImported {0} data points. \n\tData is {1} dimensional.", this.size, this.GetDataPoint(0).Length);
        }


    }
}
