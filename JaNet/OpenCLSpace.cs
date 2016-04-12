using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using OpenCL.Net;
using OpenCL.Net.Extensions;


namespace JaNet
{
#if OPENCL_ENABLED

    static class OpenCLSpace
    {

        #region Fields

        public static readonly int BASE_GROUP_SIZE = 32; // constant

        private static Context context;
        private static Device device;
        private static CommandQueue queue;

        private static List<int> maxWorkItemSizes;
        private static int maxWorkGroupSize;
        private static int maxComputeUnits;

        private static string kernelsPath;

        public static ErrorCode ClError;
        public static Event ClEvent;

        #endregion


        #region Properties

        public static Context Context
        {
            get { return context; }
            set { context = value; }
        }

        public static Device Device
        {
            get { return device; }
        }

        public static CommandQueue Queue
        {
            get { return queue; }
            set { queue = value; }
        }

        public static List<int> MaxWorkItemSizes
        {
            get { return maxWorkItemSizes; }
        }

        public static int MaxWorkGroupSize
        {
            get { return maxWorkGroupSize; }
        }

        public static int MaxComputeUnits
        {
            get { return maxComputeUnits; }
        }


        public static string KernelsPath
        {
            get { return kernelsPath; }
            set { kernelsPath = value; }
        }

        #endregion


        #region Kernels

        // Convolutional layer
        public static Kernel Im2colLookupTable;
        public static Kernel ZeroPad;
        public static Kernel ZeroUnpad;
        public static Kernel ConvForward;
        public static Kernel ConvUpdateSpeeds;
        public static Kernel ConvUpdateParameters;
        public static Kernel ConvBackPropagate;

        // Fully connected layer
        public static Kernel FCForward;
        public static Kernel FCBackward;
        public static Kernel FCUpdateSpeeds;
        public static Kernel FCUpdateParameters;

        // ReLU layer
        public static Kernel ReLUForward;
        public static Kernel ReLUBackward;

        // Softmax layer
        public static Kernel SoftmaxForward;

        // Cross-entropy gradient
        public static Kernel CrossEntropyGradient;

        // Classification
        public static Kernel CheckClassification;

        #endregion


        #region OpenCL setup

        public static void SetupSpace()
        {
            int deviceID; // will be asked to user

            List<Device> devicesList = new List<Device>();
            List<string> deviceNames = new List<string>();
            List<string> platformNames = new List<string>();
            int nDevices = 0;


            // Get list of available platforms
            Console.WriteLine("\nSearching for OpenCL-capable platforms... ");
            Platform[] platforms = Cl.GetPlatformIDs(out ClError);
            CheckErr(ClError, "CL.Setup: Cl.GetPlatformIDs");
            Console.WriteLine("{0} platforms found.\n", platforms.Length);

            foreach (Platform platform in platforms)
            {
                // Get platform info
                string platformName = Cl.GetPlatformInfo(platform, PlatformInfo.Name, out ClError).ToString();
                Console.WriteLine("Platform: " + platformName);
                CheckErr(ClError, "CL.Setup: Cl.GetPlatformInfo");

                // Get all available devices within this platform and list them on screen
                foreach (Device dev in Cl.GetDeviceIDs(platform, DeviceType.All, out ClError))
                {
                    CheckErr(ClError, "CL.Setup: Cl.GetDeviceIDs");
                    string deviceName = Cl.GetDeviceInfo(dev, DeviceInfo.Name, out ClError).ToString();
                    CheckErr(ClError, "CL.Setup: Cl.GetDeviceInfo");
                    Console.WriteLine("Device {0}: {1}", nDevices, deviceName);

                    devicesList.Add(dev);
                    deviceNames.Add(deviceName);
                    platformNames.Add(platformName);
                    nDevices++;
                }
                Console.WriteLine();
            }

            if (nDevices == 0)
            {
                throw new PlatformNotSupportedException("No OpenCL-capable platform and/or devices were found on this system.");
            }

            Console.Write("Enter ID of device to use: ");
            deviceID = int.Parse(Console.ReadLine());

            // Select device according to user's choice
            device = devicesList[deviceID];
            Console.WriteLine("\nUsing device {0}", deviceNames[deviceID]);

            // Create OpenCL context
            context = Cl.CreateContext(null, 1, new[] { device }, ContextNotify, IntPtr.Zero, out ClError);    //Second parameter is amount of devices
            CheckErr(ClError, "CL.Setup: Cl.CreateContext");

            // Create OpenCL command queue
            queue = Cl.CreateCommandQueue(context, device, (CommandQueueProperties)0, out ClError);
            CheckErr(ClError, "CL.Setup: Cl.CreateCommandQueue");

            // Extract some device info
            maxWorkItemSizes = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemSizes, out ClError).CastToEnumerable<int>(new int[] { 0, 1, 2 }).ToList();
            Console.WriteLine("Max work item sizes: {0}, {1}, {2}", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

            maxWorkGroupSize = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, out ClError).CastTo<int>();
            Console.WriteLine("Max work group size: {0}", maxWorkGroupSize);

            maxComputeUnits = Cl.GetDeviceInfo(device, DeviceInfo.MaxComputeUnits, out ClError).CastTo<int>();
            Console.WriteLine("Max compute units: {0}", maxComputeUnits);
        }


        #endregion


        #region Kernel loading and building

        public static Kernel LoadAndBuildKernel(string kernelFilePath, string kernelName)
        {
            // Attempt to read file
            if (!System.IO.File.Exists(kernelFilePath))
            {
                Console.WriteLine("Program doesn't exist at path " + kernelFilePath);
                Console.ReadKey();
                System.Environment.Exit(1);
            }
            string kernelSource = System.IO.File.ReadAllText(kernelFilePath);

            // Create program
            OpenCL.Net.Program clProgram = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out ClError);
            CheckErr(ClError, "CL.LoadAndBuildKernel: Cl.CreateProgramWithSource");

            //Compile kernel source
            ClError = Cl.BuildProgram(clProgram, 1, new[] { device }, string.Empty, null, IntPtr.Zero);
            CheckErr(ClError, "CL.LoadAndBuildKernel: Cl.BuildProgram " + kernelName);

            //Check for any compilation errors
            if (Cl.GetProgramBuildInfo(clProgram, device, ProgramBuildInfo.Status, out ClError).CastTo<BuildStatus>()
                != BuildStatus.Success)
            {
                CheckErr(ClError, "CL.LoadAndBuildKernel: Cl.GetProgramBuildInfo");
                Console.WriteLine("Cl.GetProgramBuildInfo != Success");
                Console.WriteLine(Cl.GetProgramBuildInfo(clProgram, device, ProgramBuildInfo.Log, out ClError));
                Console.ReadKey();
                System.Environment.Exit(1);
            }
            //Create the required kernel (entry function)
            Kernel kernel = Cl.CreateKernel(clProgram, kernelName, out ClError);
            CheckErr(ClError, "CL.LoadAndBuildKernel: Cl.CreateKernel " + kernelName);

            return kernel;
        }


        public static void LoadKernels()
        {
            if (kernelsPath == null)
                throw new MissingFieldException("Path to kernels' source must be specified before calling LoadKernels()");

            // Convolutional layer
            Im2colLookupTable = LoadAndBuildKernel(kernelsPath + "/Im2colLookupTable.cl", "Im2colLookupTable");
            ZeroPad = LoadAndBuildKernel(kernelsPath + "/ZeroPad.cl", "ZeroPad");
            ZeroUnpad = LoadAndBuildKernel(kernelsPath + "/ZeroUnpad.cl", "ZeroUnpad");
            ConvForward = LoadAndBuildKernel(kernelsPath + "/ConvForward.cl", "ConvForward");
            ConvUpdateSpeeds = LoadAndBuildKernel(kernelsPath + "/ConvUpdateSpeeds.cl", "ConvUpdateSpeeds");
            ConvUpdateParameters = LoadAndBuildKernel(kernelsPath + "/ConvUpdateParameters.cl", "ConvUpdateParameters");
            ConvBackPropagate = LoadAndBuildKernel(kernelsPath + "/ConvBackPropagate.cl", "ConvBackPropagate");

            // Fully connected layer
            FCForward = LoadAndBuildKernel(kernelsPath + "/FCForward.cl", "FCForward");
            FCBackward = LoadAndBuildKernel(kernelsPath + "/FCBackward.cl", "FCBackward");
            FCUpdateSpeeds = LoadAndBuildKernel(kernelsPath + "/FCUpdateSpeeds.cl", "FCUpdateSpeeds");
            FCUpdateParameters = LoadAndBuildKernel(kernelsPath + "/FCUpdateParameters.cl", "FCUpdateParameters");

            // ReLU layer
            ReLUForward = LoadAndBuildKernel(kernelsPath + "/ReLUForward.cl", "ReLUForward");
            ReLUBackward = LoadAndBuildKernel(kernelsPath + "/ReLUBackward.cl", "ReLUBackward");

            // Softmax layer
            SoftmaxForward = LoadAndBuildKernel(kernelsPath + "/SoftmaxForward.cl", "SoftmaxForward");

            // Cross-entropy gradient
            CrossEntropyGradient = LoadAndBuildKernel(kernelsPath + "/CrossEntropyGradient.cl", "CrossEntropyGradient");


            // Classification
            // TODO: implement a better kernel
            CheckClassification = LoadAndBuildKernel(kernelsPath + "/CheckClassification.cl", "CheckClassification");
        }


        #endregion


        #region Helper methods

        public static void WipeBuffer(Mem buffer, int nElementsInBuffer, Type type)
        {
            float[] zeros = new float[nElementsInBuffer];
            int sizeOfElement;

            if (type == typeof(float))
                sizeOfElement = sizeof(float);
            else if (type == typeof(int))
                sizeOfElement = sizeof(int);
            else
                throw new ArgumentException("Type not supported. Use either float or int.");

            ClError = Cl.EnqueueWriteBuffer(    queue,
                                                buffer,
                                                OpenCL.Net.Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)(sizeOfElement * nElementsInBuffer),
                                                zeros,
                                                0,
                                                null,
                                                out ClEvent);
            CheckErr(ClError, "OpenCLSpace.WipeBuffer: Cl.EnqueueWriteBuffer");
        }
        
        public static void CheckErr(ErrorCode err, string name)
        {
            if (err != ErrorCode.Success)
            {
                Console.WriteLine("ERROR: " + name + " (" + err.ToString() + ")");
                Console.ReadKey();
            }
        }

        public static void ContextNotify(string errInfo, byte[] data, IntPtr cb,
            IntPtr userData)
        {
            Console.WriteLine("OpenCL Notification: " + errInfo);
        }

        #endregion

    }

#endif


    // OLD
    /*
#if OPENCL_ENABLED
    static class CL
    {

        #region CL helper class fields

        private static Context context;
        private static Device device;
        private static CommandQueue queue;

        private static List<int> maxWorkItemSizes;
        private static int maxWorkGroupSize;
        private static int maxComputeUnits;

        public static ErrorCode Error;
        public static Event Event;

        private static string kernelsPath;
        #endregion


        #region CL helper class properties (public)

        public static Context Context
        {
            get { return context; }
            set { context = value; }
        }

        public static Device Device
        {
            get { return device; }
        }

        public static CommandQueue Queue
        {
            get { return queue; }
            set { queue = value; }
        }

        public static List<int> MaxWorkItemSizes
        {
            get { return maxWorkItemSizes; }
        }

        public static int MaxWorkGroupSize
        {
            get { return maxWorkGroupSize; }
        }

        public static int MaxComputeUnits
        {
            get { return maxComputeUnits; }
        }

        public static string KernelsPath
        {
            get { return kernelsPath; }
        }
        #endregion


        #region Kernels (public)

        // FullyConnected layer
        //public static Kernel FCForward;
        //public static Kernel FCBackward;
        //public static Kernel FCUpdateParameters;

        // ReLU layer
        //public static Kernel ReLUForward;
        //public static Kernel ReLUBackward;

        // Softmax layer
        //public static Kernel SoftmaxForward;

        // Cross-entropy gradient
        //public static Kernel CrossEntropyGradient;

        // Classification
        public static Kernel CheckClassification;

        #endregion


        #region OpenCL setup and finalization

        public static void Setup(string KernelsPath)
        {
            Console.WriteLine("\n=========================================");
            Console.WriteLine("    OpenCL setup");
            Console.WriteLine("=========================================\n");
            
            int deviceID; // will be asked to user

            List<Device> devicesList = new List<Device>();
            List<string> deviceNames = new List<string>();
            List<string> platformNames = new List<string>();
            int nDevices = 0;

            // Get list of available platforms
            Console.WriteLine("\nSearching for OpenCL-capable platforms... ");
            Platform[] platforms = Cl.GetPlatformIDs(out Error);
            CheckErr(Error, "CL.Setup: Cl.GetPlatformIDs");
            Console.WriteLine("{0} platforms found.\n", platforms.Length);

            //Console.WriteLine("\n=============================================\n");
            foreach (Platform platform in platforms)
            {
                string platformName = Cl.GetPlatformInfo(platform, PlatformInfo.Name, out Error).ToString();
                Console.WriteLine("Platform: " + platformName);
                CheckErr(Error, "CL.Setup: Cl.GetPlatformInfo");

                // Get all available devices for this platform and list them on screen
                foreach (Device dev in Cl.GetDeviceIDs(platform, DeviceType.All, out Error))
                {
                    CheckErr(Error, "CL.Setup: Cl.GetDeviceIDs");
                    string deviceName = Cl.GetDeviceInfo(dev, DeviceInfo.Name, out Error).ToString();
                    CheckErr(Error, "CL.Setup: Cl.GetDeviceInfo");
                    Console.WriteLine("Device {0}: {1}", nDevices, deviceName);

                    devicesList.Add(dev);
                    deviceNames.Add(deviceName);
                    platformNames.Add(platformName);
                    nDevices++;
                }
                Console.WriteLine();
            }

            if (nDevices == 0)
            {
                Console.WriteLine("FATAL ERROR: No devices found!");
                return;
            }

            Console.Write("Enter ID of device to use: ");
            deviceID = int.Parse(Console.ReadLine()); 

            // Select device according to user's choice
            device = devicesList[deviceID];
            Console.WriteLine("\nUsing device {0}", deviceNames[deviceID]);

            // Create OpenCL context
            context = Cl.CreateContext(null, 1, new[] { device }, ContextNotify, IntPtr.Zero, out Error);    //Second parameter is amount of devices
            CheckErr(Error, "CL.Setup: Cl.CreateContext");

            // Create OpenCL command queue
            queue = Cl.CreateCommandQueue(context, device, (CommandQueueProperties)0, out Error);
            CheckErr(Error, "CL.Setup: Cl.CreateCommandQueue");

            // Extract some device info
            maxWorkItemSizes = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkItemSizes, out Error).CastToEnumerable<int>(new int[] { 0, 1, 2 }).ToList();
            Console.WriteLine("Max work item sizes: {0}, {1}, {2}", maxWorkItemSizes[0], maxWorkItemSizes[1], maxWorkItemSizes[2]);

            maxWorkGroupSize = Cl.GetDeviceInfo(device, DeviceInfo.MaxWorkGroupSize, out Error).CastTo<int>();
            Console.WriteLine("Max work group size: {0}", maxWorkGroupSize);

            maxComputeUnits = Cl.GetDeviceInfo(device, DeviceInfo.MaxComputeUnits, out Error).CastTo<int>();
            Console.WriteLine("Max compute units: {0}", maxComputeUnits);

            // Set kernel path
            kernelsPath = KernelsPath;
        }


        #endregion


        #region Kernel loading and building

        public static Kernel LoadBuildKernel(string kernelFilePath, string kernelName)
        {

            // Attempt to read file
            if (!System.IO.File.Exists(kernelFilePath))
            {
                Console.WriteLine("Program doesn't exist at path " + kernelFilePath);
                Console.ReadKey();
                System.Environment.Exit(1);
            }
            string kernelSource = System.IO.File.ReadAllText(kernelFilePath);

            // Create program
            OpenCL.Net.Program clProgram = Cl.CreateProgramWithSource(context, 1, new[] { kernelSource }, null, out Error);
            CheckErr(Error, "CL.LoadAndBuildKernel: Cl.CreateProgramWithSource");

            //Compile kernel source
            Error = Cl.BuildProgram(clProgram, 1, new[] { device }, string.Empty, null, IntPtr.Zero);
            CheckErr(Error, "CL.LoadAndBuildKernel: Cl.BuildProgram");

            //Check for any compilation errors
            if (Cl.GetProgramBuildInfo(clProgram, device, ProgramBuildInfo.Status, out Error).CastTo<BuildStatus>()
                != BuildStatus.Success)
            {
                CheckErr(Error, "CL.LoadAndBuildKernel: Cl.GetProgramBuildInfo");
                Console.WriteLine("Cl.GetProgramBuildInfo != Success");
                Console.WriteLine(Cl.GetProgramBuildInfo(clProgram, device, ProgramBuildInfo.Log, out Error));
                Console.ReadKey();
                System.Environment.Exit(1);
            }
            //Create the required kernel (entry function)
            Kernel kernel = Cl.CreateKernel(clProgram, kernelName, out Error);
            CheckErr(Error, "CL.LoadAndBuildKernel: Cl.CreateKernel");

            return kernel;
        }


        public static void LoadKernels(string kernelsPath)
        {
            // Classification

            string classificationName = "CheckClassification";
            CheckClassification = LoadBuildKernel(kernelsPath + "/" + classificationName + ".cl", classificationName);
        }

        #endregion


        #region Helper methods

        public static void CheckErr(ErrorCode err, string name)
        {
            if (err != ErrorCode.Success)
            {
                Console.WriteLine("ERROR: " + name + " (" + err.ToString() + ")");
                Console.ReadKey();
            }
        }

        public static void ContextNotify(string errInfo, byte[] data, IntPtr cb,
            IntPtr userData)
        {
            Console.WriteLine("OpenCL Notification: " + errInfo);
        }

        public static void ClearBuffer(Mem buffer, int bufferSize)
        {
            float[] zeros = new float[bufferSize];
            CL.Error = Cl.EnqueueWriteBuffer(   CL.Queue,
                                                buffer,
                                                OpenCL.Net.Bool.True,
                                                (IntPtr)0,
                                                (IntPtr)(sizeof(float) * bufferSize),
                                                zeros,
                                                0,
                                                null,
                                                out CL.Event);
            CL.CheckErr(CL.Error, "CL.ClearBuffer: Cl.EnqueueWriteBuffer");
        }

        #endregion

    }

#endif
     
     */ 
}
