# DCGAN on CelebA - Google Colab Implementation

## 📌 Project Overview
This project implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate realistic human faces using the **CelebA** dataset. The entire implementation is optimized for **Google Colab** and utilizes **PyTorch** for model training and image generation.

## 🚀 Features
✅ **Loads and preprocesses the CelebA dataset** in Google Colab without requiring Google Drive.  
✅ **Implements DCGAN (Generator & Discriminator)** following Radford et al.'s 2015 paper.  
✅ **Uses PyTorch** for easy training and deployment.  
✅ **Saves & loads model checkpoints** for better control over training.  
✅ **Visualizes generated images** during training to track model performance.  

---

## 📥 Dataset Download & Setup
The **CelebA dataset** is downloaded automatically via Kaggle API.

### **Step 1: Get Kaggle API Key**
1. Go to [Kaggle](https://www.kaggle.com/)
2. Click on your profile picture → **Account**
3. Scroll down to **API** → Click **Create New API Token**
4. A file named `kaggle.json` will be downloaded.

### **Step 2: Upload `kaggle.json` to Colab**
Run the following code in a **Colab cell**, then manually upload the `kaggle.json` file.
```python
from google.colab import files
files.upload()  # Upload `kaggle.json`
```

### **Step 3: Download CelebA Dataset**
```python
!mkdir -p ~/.kaggle
!mv kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d jessicali9530/celeba-dataset
!unzip -q celeba-dataset.zip -d /content/CelebA
```

---

## 🛠 Model Implementation
### **1️⃣ Generator (G)**
The generator takes a **random noise vector (latent space, `z`)** and generates a **64×64 image** using **transposed convolution layers** with batch normalization and ReLU activation.

```python
class Generator(nn.Module):
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
```

### **2️⃣ Discriminator (D)**
The discriminator is a **binary classifier** that determines if an image is real or fake using **convolutional layers** with **LeakyReLU activations**.

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
```

---

## 🎯 Training the Model
Run the training script:
```python
python train.py
```
The loss values for the generator and discriminator will be printed every few iterations.

---

## 📷 Generating Images
Once the model is trained, use the following command to generate and visualize fake images:
```python
python generate.py
```
Alternatively, in Colab, run:
```python
def generate_images():
    gen.eval()
    noise = torch.randn(64, 100, 1, 1, device=device)
    with torch.no_grad():
        fake_images = gen(noise).cpu()
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.imshow(np.transpose(vutils.make_grid(fake_images, padding=2, normalize=True), (1,2,0)))
    plt.show()

generate_images()
```

---

## 🛠 Dependencies
Ensure you have the required libraries installed:
```bash
pip install torch torchvision numpy matplotlib tqdm kaggle
```

---

## 🔥 Results & Improvements
- **Expected Output:** The model generates **realistic human faces** after training for 50+ epochs.
- **Possible Improvements:**
  - Train for **more epochs (100-200)**.
  - Use **higher resolution images (128x128 or 256x256)**.
  - Experiment with **different optimizers** (e.g., RMSprop).

---

## 📜 References
- Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*.
- Official [CelebA Dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)

---

## 🎯 Final Notes
✅ This DCGAN implementation is **Colab-friendly**.  
✅ Works **without Google Drive** (entirely in Colab's storage).  
✅ Provides **high-quality image generation** with minimal effort!  

🔥 **Happy Training!** 🚀

