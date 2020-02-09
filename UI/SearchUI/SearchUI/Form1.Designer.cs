namespace SearchUI
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.browserbtn = new System.Windows.Forms.Button();
            this.load_files_txtbox = new System.Windows.Forms.TextBox();
            this.OKbtn = new System.Windows.Forms.Button();
            this.cancelbtn = new System.Windows.Forms.Button();
            this.openFileDialog1 = new System.Windows.Forms.OpenFileDialog();
            this.SuspendLayout();
            // 
            // browserbtn
            // 
            this.browserbtn.BackColor = System.Drawing.SystemColors.HighlightText;
            this.browserbtn.ImageAlign = System.Drawing.ContentAlignment.TopRight;
            this.browserbtn.Location = new System.Drawing.Point(416, 80);
            this.browserbtn.Name = "browserbtn";
            this.browserbtn.Size = new System.Drawing.Size(105, 29);
            this.browserbtn.TabIndex = 0;
            this.browserbtn.Text = "Browser";
            this.browserbtn.UseVisualStyleBackColor = false;
            this.browserbtn.Click += new System.EventHandler(this.browserbtn_Click);
            // 
            // load_files_txtbox
            // 
            this.load_files_txtbox.Location = new System.Drawing.Point(60, 80);
            this.load_files_txtbox.Name = "load_files_txtbox";
            this.load_files_txtbox.Size = new System.Drawing.Size(327, 29);
            this.load_files_txtbox.TabIndex = 1;
            // 
            // OKbtn
            // 
            this.OKbtn.BackColor = System.Drawing.SystemColors.HighlightText;
            this.OKbtn.ImageAlign = System.Drawing.ContentAlignment.TopRight;
            this.OKbtn.Location = new System.Drawing.Point(121, 296);
            this.OKbtn.Name = "OKbtn";
            this.OKbtn.Size = new System.Drawing.Size(128, 50);
            this.OKbtn.TabIndex = 2;
            this.OKbtn.Text = "OK";
            this.OKbtn.UseVisualStyleBackColor = false;
            this.OKbtn.Click += new System.EventHandler(this.OKbtn_Click);
            // 
            // cancelbtn
            // 
            this.cancelbtn.BackColor = System.Drawing.SystemColors.HighlightText;
            this.cancelbtn.ImageAlign = System.Drawing.ContentAlignment.TopRight;
            this.cancelbtn.Location = new System.Drawing.Point(291, 296);
            this.cancelbtn.Name = "cancelbtn";
            this.cancelbtn.Size = new System.Drawing.Size(128, 50);
            this.cancelbtn.TabIndex = 3;
            this.cancelbtn.Text = "Cancel";
            this.cancelbtn.UseVisualStyleBackColor = false;
            // 
            // openFileDialog1
            // 
            this.openFileDialog1.FileName = "openFileDialog1";
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(11F, 24F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.BackColor = System.Drawing.SystemColors.ControlLight;
            this.ClientSize = new System.Drawing.Size(584, 411);
            this.Controls.Add(this.cancelbtn);
            this.Controls.Add(this.OKbtn);
            this.Controls.Add(this.load_files_txtbox);
            this.Controls.Add(this.browserbtn);
            this.Font = new System.Drawing.Font("Microsoft Sans Serif", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.Margin = new System.Windows.Forms.Padding(6);
            this.Name = "Form1";
            this.Text = "Form1";
            this.Load += new System.EventHandler(this.Form1_Load);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.Button browserbtn;
        private System.Windows.Forms.TextBox load_files_txtbox;
        private System.Windows.Forms.Button OKbtn;
        private System.Windows.Forms.Button cancelbtn;
        private System.Windows.Forms.OpenFileDialog openFileDialog1;
    }
}

