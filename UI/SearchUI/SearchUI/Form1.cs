using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using OpenCvSharp;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;


namespace SearchUI
{
    public partial class Form1 : Form
    {
        Mat inputimg;
        const int IMG_W = 32, IMG_H = 32;
        private void Form1_Load(object sender, EventArgs e)
        {

        }
        public Form1()
        {
            InitializeComponent();
        }

        static float[] LoadTensorFromFile(ref Mat img)
        {
            //Input image to buffer
            int size = (int)img.Total() * img.ElemSize();
            Console.WriteLine("size {0}", size);
            float[] tensorData = new float[size];
            //img.ToBytes().Clone(tensorData);

            return  tensorData;
        }


        private bool RunONNX(string path, ref Mat img)
        {
            try
            {
                //string basepath = path/*"..\\..\\..\\testdata\\"*/;
                string modelPath = path + "model.onnx";
                // Optional : Create session options and set the graph optimization level for the session
                SessionOptions options = new SessionOptions();
                options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_EXTENDED;

                using (var session = new InferenceSession(modelPath, options))
                {
                    var inputMeta = session.InputMetadata;
                    var container = new List<NamedOnnxValue>();

                    float[] inputData = LoadTensorFromFile(ref img); // this is the data for only one input tensor for this model
                
                    foreach (var name in inputMeta.Keys)
                    {
      
                        var tensor = new DenseTensor<float>(inputData, inputMeta[name].Dimensions);
                        container.Add(NamedOnnxValue.CreateFromTensor<float>(name, tensor));
                    }

                    // Run the inference
                    using (var results = session.Run(container))  // results is an IDisposableReadOnlyCollection<DisposableNamedOnnxValue> container
                    {
                        // dump the results
                        foreach (var r in results)
                        {
                            Console.WriteLine("Output for {0}", r.Name);
                            Console.WriteLine(r.AsEnumerable<float>().ToString());
                        }
                    }
                }
            }
            catch(Exception ex)
            {
                Console.WriteLine("inputMeta.Keys {0}", ex.ToString());
                return false;
            }

            return true;
        }

        void preprocess(ref Mat img)
        {
            if(img.Cols > IMG_W || img.Rows > IMG_H)
            {

                OpenCvSharp.Size size;
                size.Height = IMG_H;
                size.Width = IMG_W;
                Cv2.Resize(img, img, size);
            }
        }

        private FileExtension checkimgfile(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open, FileAccess.Read);
            System.IO.BinaryReader br = new System.IO.BinaryReader(fs);
            FileExtension extension;
            string fileType = string.Empty; ;
            try
            {
                byte data = br.ReadByte();
                fileType += data.ToString();
                data = br.ReadByte();
                fileType += data.ToString();
                
                
                extension = (FileExtension)Enum.Parse(typeof(FileExtension), fileType);

         
                switch (extension)
                {
                    case FileExtension.GIF:
                    case FileExtension.JPG:
                    case FileExtension.PNG:
                        break;
                    default:
                        extension = FileExtension.VALIDFILE;
                        break;
                }
            }
            catch (Exception ex)
            {
                throw ex;
            }
            
                if (fs != null)
                {
                    fs.Close();
                    br.Close();
                }

            return extension;
        }


        public enum FileExtension
        {
            JPG = 255216,
            GIF = 7173,
            PNG = 13780,
            VALIDFILE = 9999999
        }
        private void browserbtn_Click(object sender, EventArgs e)
        {
            FileExtension extension;
            openFileDialog1.Title = "C# Corner Open File Dialog";
            openFileDialog1.InitialDirectory = @"c:\";
            openFileDialog1.Filter = "All files (*.*)|*.*|All files (*.*)|*.*";
            openFileDialog1.FilterIndex = 2;
            openFileDialog1.RestoreDirectory = true;

            if (openFileDialog1.ShowDialog() == DialogResult.OK)
            {
                load_files_txtbox.Text = openFileDialog1.FileName;
                /*var ext = openFileDialog1.FileName.Substring(openFileDialog1.FileName.LastIndexOf(".") + 1,
                    openFileDialog1.FileName.Length - openFileDialog1.FileName.LastIndexOf(".") - 1); Get extension name*/

                extension = checkimgfile(load_files_txtbox.Text);
                if (extension == FileExtension.VALIDFILE)
                {
                    MessageBox.Show("VALIDFILE", "Error");
                    load_files_txtbox.Clear();
                }
                else
                {

                    inputimg = Cv2.ImRead(load_files_txtbox.Text);
                    preprocess(ref inputimg);
                    /*Cv2.ImShow("A", img);
                    Cv2.WaitKey(0);
                    Cv2.DestroyAllWindows();*/

                }
            }

        }

        private void OKbtn_Click(object sender, EventArgs e)
        {
            if(!RunONNX("..\\..\\..\\..\\..\\..\\", ref inputimg))
                MessageBox.Show("VALIDFILE OF RUNNING ONNX", "Error");
            else
                MessageBox.Show("Succeed to load", "Good");

            load_files_txtbox.Clear();
        }
    }
}
