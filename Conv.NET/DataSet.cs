using OpenCL.Net;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Globalization;
using System.IO;

namespace Conv.NET
{
    [Serializable]
    public class DataSet
    {
        #region Publicly accessible data
        public int DataDimension { get; private set; }

        public int NumberOfClasses { get; private set; }

        public List<DataItem> DataContainer { get; private set; }
        #endregion

        #region Constructor
        /// <summary>
        /// A class for storing Data and Labels
        /// </summary>
        /// <param name="nClasses">The number of classes in the network</param>
        public DataSet(int nClasses)
        {
            NumberOfClasses = nClasses;
            DataContainer = new List<DataItem>();
        }
        #endregion

        #region Methods
        public void ReadData(string dataPath, string labelsPath)
        {
            string[] dataArray = File.ReadAllLines(dataPath);
            string[] labelsArray = File.ReadAllLines(labelsPath);

            if (dataArray.Length != labelsArray.Length)
            {
                throw new Exception("The amount of data does not match the amount of labels");
            }

            // Read images and their labels
            for (int index = 0; index < dataArray.Length; index++)
            {
                string[] columns = dataArray[index].Split('\t');

                DataDimension = columns.Length;

#if OPENCL_ENABLED
                float[] dataPoint = new float[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    dataPoint[i] = float.Parse(columns[i], CultureInfo.InvariantCulture.NumberFormat);
                }

                int datumBytesSize = sizeof(float) * dataPoint.Length;
                Mem tmpBuffer = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                            MemFlags.ReadOnly | MemFlags.CopyHostPtr | MemFlags.AllocHostPtr,
                                                            (IntPtr)datumBytesSize,
                                                            dataPoint,
                                                            out OpenCLSpace.ClError);
                OpenCLSpace.CheckErr(OpenCLSpace.ClError, "DataSet(): Cl.CreateBuffer tmpBuffer");
#else
                double[] tmpBuffer = new double[columns.Length];
                for (int i = 0; i < columns.Length; i++)
                {
                    tmpBuffer[i] = double.Parse(columns[i], CultureInfo.InvariantCulture.NumberFormat);
                }
#endif

                DataContainer.Add(new DataItem(tmpBuffer, Convert.ToInt32(labelsArray[index])));
            }
        }

        public void ReadImage(Image input, int label)
        {
            unsafe
            {
                using (Bitmap bmp = new Bitmap(input))
                {
                    int offSet = bmp.Width * bmp.Height;
                    DataDimension = offSet * 3;
#if OPENCL_ENABLED
                    float[] dataPoint = new float[DataDimension];
#else
                    double[] dataPoint = new double[DataDimension];
#endif
                    #region Copy RGB values directly from memory to the array
                    BitmapData bitmapData = bmp.LockBits(new Rectangle(0, 0, bmp.Width, bmp.Height), ImageLockMode.ReadOnly, bmp.PixelFormat);
                    int bytesPerPixel = Image.GetPixelFormatSize(bmp.PixelFormat) / 8;
                    int heightInPixels = bitmapData.Height;
                    int widthInBytes = bitmapData.Width * bytesPerPixel;
                    byte* ptrFirstPixel = (byte*)bitmapData.Scan0;

                    int index = 0;
                    for (int y = 0; y < heightInPixels; y++)
                    {
                        byte* currentLine = ptrFirstPixel + (y * bitmapData.Stride);
                        for (int x = 0; x < widthInBytes; x = x + bytesPerPixel)
                        {
                            dataPoint[index] = currentLine[x + 2]; // Red
                            dataPoint[index + offSet] = currentLine[x + 1]; // Green
                            dataPoint[index + offSet + offSet] = currentLine[x]; // Blue
                            index++;
                        }
                    }

                    bmp.UnlockBits(bitmapData);
                    #endregion

#if OPENCL_ENABLED
                    int datumBytesSize = sizeof(float) * dataPoint.Length;
                    Mem tmpBuffer = (Mem)Cl.CreateBuffer(OpenCLSpace.Context,
                                                                MemFlags.ReadOnly | MemFlags.CopyHostPtr | MemFlags.AllocHostPtr,
                                                                (IntPtr)datumBytesSize,
                                                                dataPoint,
                                                                out OpenCLSpace.ClError);
                    OpenCLSpace.CheckErr(OpenCLSpace.ClError, "DataSet(): Cl.CreateBuffer tmpBuffer");

                    DataContainer.Add(new DataItem(tmpBuffer, label));
#else
                    DataContainer.Add(new DataItem(dataPoint, label));
#endif
                }
            }
        }
        #endregion
    }

    #region DataItem
    public class DataItem
    {
        public int Label { get; private set; }

#if OPENCL_ENABLED
        public Mem Data { get; private set; }

        public DataItem(Mem dataInput, int label)
        {
            Data = dataInput;
            Label = label;
        }
#else
        public double[] Data { get; private set; }

        public DataItem(double[] dataInput, int label)
        {
            Data = dataInput;
            Label = label;
        }
#endif
    }
    #endregion
}
