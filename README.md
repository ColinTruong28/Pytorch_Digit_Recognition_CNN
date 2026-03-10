# Handwritten Digit Recognition CNN using PyTorch

My Convolutional Neural Network is sequential consisting of a convolutional and a fully connected layer followed by a fully connected classification head, designed to process 28x28 grayscale images into 10 digit classes.

# Results | Best validation accuracy: 0.9760

# Confusion matrix for best checkpoint:
$$
\begin{pmatrix}
50 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 50 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 48 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 49 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 50 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 49 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 49 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 50 & 0 & 0 \\
0 & 0 & 1 & 2 & 0 & 1 & 0 & 0 & 46 & 0 \\
1 & 1 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 47 
\end{pmatrix}
$$

# Heat Map Visualization

<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/28eb1526-102b-40b3-b499-9a19887e9195" />
