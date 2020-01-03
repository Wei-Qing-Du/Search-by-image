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

namespace SearchUI
{
    public partial class Form1 : Form
    {

        private void Form1_Load(object sender, EventArgs e)
        {

        }
        public Form1()
        {
            InitializeComponent();
        }
        private FileExtension checkimgfile(string fileName)
        {
            FileStream fs = new FileStream(fileName, FileMode.Open, FileAccess.Read);
            System.IO.BinaryReader br = new System.IO.BinaryReader(fs);
            string fileType = string.Empty; ;
            try
            {
                byte data = br.ReadByte();
                fileType += data.ToString();
                data = br.ReadByte();
                fileType += data.ToString();
                FileExtension extension;
                
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
       
                return extension;
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
                textBox1.Text = openFileDialog1.FileName;
                /*var ext = openFileDialog1.FileName.Substring(openFileDialog1.FileName.LastIndexOf(".") + 1,
                    openFileDialog1.FileName.Length - openFileDialog1.FileName.LastIndexOf(".") - 1); Get extension name*/

                extension = checkimgfile(textBox1.Text);
                if (extension == FileExtension.VALIDFILE)
                {
                    MessageBox.Show("VALIDFILE", "Error");
                    textBox1.Clear();
                }
                else
                {
                    #if DEBUG
                        Mat img = Cv2.ImRead(textBox1.Text);
                        Cv2.ImShow("A", img);
                        Cv2.WaitKey(0);
                        Cv2.DestroyAllWindows();
                    #endif
                }
            }

        }
    }
}
