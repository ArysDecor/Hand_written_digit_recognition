```markdown
#  Handwritten Digit Recognition with CNN and PyTorch

This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify handwritten digits from the **MNIST** dataset.  
It also provides an **interactive drawing interface (Tkinter GUI)** where you can draw your own digits and get real-time predictions.

---

##  Project Overview

- Dataset: **MNIST** (28x28 grayscale images of digits 0–9)
- Framework: **PyTorch**
- Model: **CNN (Convolutional Neural Network)**
- GUI: **Tkinter + Pillow**

---

##  Demo

![demo](./demo.gif)  
> Draw a digit → Predict → Result shown in real-time!

---

##  Folder Structure

```

 HandwrittenDigitRecognition/
│
├── mnist\_cnn.pth           # Trained model weights
├── main.py                 # GUI + Prediction code
├── train.py                # CNN training code
├── README.md               # This file

````

---

##  How CNN Works (Mathematical Intuition)

CNNs (Convolutional Neural Networks) are specialized for **image classification**. Here’s a mathematical breakdown of how they process input:

### 1.  Input Image

- An MNIST digit is a **28×28 grayscale** image.
- Each pixel is normalized in the range [0, 1].

Let \( X \in \mathbb{R}^{28 \times 28} \)

---

### 2.  Convolution Layer

Applies a **kernel/filter** \( K \in \mathbb{R}^{3 \times 3} \), sliding over the image and computing:

\[
Y_{i,j} = \sum_{m=0}^{2} \sum_{n=0}^{2} X_{i+m,j+n} \cdot K_{m,n}
\]

- Result: A **feature map** highlighting edges or patterns.
- With multiple filters, we get multiple channels.

---

### 3. 🎚 Activation Function (ReLU)

Applies a **non-linearity** to introduce complexity:

\[
f(x) = \max(0, x)
\]

- Ensures the model can learn complex functions.

---

### 4.  Max Pooling

Reduces spatial dimensions by taking the max value in each 2×2 window:

\[
Y_{i,j} = \max \{ X_{2i,2j}, X_{2i,2j+1}, X_{2i+1,2j}, X_{2i+1,2j+1} \}
\]

- This improves **translation invariance** and **reduces overfitting**.

---

### 5.  Fully Connected Layer

After flattening the image, we apply:

\[
\text{output} = W \cdot x + b
\]

- Transforms learned features into final class scores.

---

### 6.  Softmax for Classification

Returns a probability vector over 10 classes (0–9):

\[
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{10} e^{z_j}}
\]

- The highest probability indicates the predicted digit.

---

## 🖥 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/handwritten-digit-cnn.git
cd handwritten-digit-cnn
````

### 2. Install requirements

```bash
pip install torch torchvision matplotlib pillow
```

### 3. Train the model (optional)

```bash
python train.py
```

> The script saves the trained weights as `mnist_cnn.pth`

### 4. Run the GUI app

```bash
python main.py
```

---

##  GUI Features

* Draw a digit using your mouse.
* Click **"Predict"** to see the CNN's prediction.
* Click **"Clear"** to draw again.

---

##  Model Architecture

| Layer             | Type           | Output Shape |
| ----------------- | -------------- | ------------ |
| Input             | 1 × 28 × 28    |              |
| Conv2d (1→32)     | 3×3 kernel     | 32 × 26 × 26 |
| ReLU              | Activation     |              |
| Conv2d (32→64)    | 3×3 kernel     | 64 × 24 × 24 |
| MaxPooling2d      | 2×2            | 64 × 12 × 12 |
| Dropout (0.25)    | Regularization |              |
| Flatten           | —              | 9216         |
| Linear (9216→128) | Dense Layer    | 128          |
| ReLU              |                |              |
| Linear (128→10)   | Output Layer   | 10 classes   |

---

##  Accuracy

After 5 epochs of training on MNIST:

* **Test Accuracy**: \~98%
* **Training Time**: \~1–2 minutes (on CPU)

---

##  Credits

* PyTorch
* MNIST Dataset
* Tkinter & PIL

---

##  License

This project is licensed under the MIT License.
