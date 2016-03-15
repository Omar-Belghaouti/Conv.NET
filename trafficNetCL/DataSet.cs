using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using System.Threading.Tasks;

namespace TrafficNetCL
{
    class DataSet
    {

        private List<float[]> trafficSigns;
        private List<int> labels;
        private int size;

        public void Add(float[] DataPoint, int Class)
        {
            this.trafficSigns.Add(DataPoint);
            this.labels.Add(Class);
            this.size += 1;
        }

        public float[] TrafficSign(int Index)
        {
            return this.trafficSigns[Index];
        }
        public int Label(int Index)
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

            // Initialize empty lists
            this.trafficSigns = new List<float[]>();
            this.labels = new List<int>();
            this.size = 0;

            foreach (var line in System.IO.File.ReadAllLines(dataPath))
            {
                var columns = line.Split('\t');

                Console.WriteLine("Line {0} reads: {1}", this.size, line);

                float[] DataPoint = new float[columns.Length - 1];
                for (int i = 0; i < columns.Length - 2; i++)
                    DataPoint[i] = Convert.ToSingle(columns[i].Trim());

                this.Add(DataPoint, Convert.ToInt32(columns[columns.Length - 1].Trim()));
            }

            Console.WriteLine("\tImported {0} data points. \n\tData is {1} dimensional.", this.size, this.TrafficSign(0).Length);
        }


    }
}
