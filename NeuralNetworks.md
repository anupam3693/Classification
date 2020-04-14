###Simplistic way to understand Neural Networks learning step by step from initialization to prediction

<img src="/Users/anupam7936/afb/Blogs/image-Neural1.png" alt="image-20200414070040022" style="zoom:50%;" />



Consider the above simplistic form of neural network(NN). This network can learn to classify as well regress the output variable referred as y also called as target variable. We use NN whenever there is a non-linear relationship between the input features (x) and the output variable (y). This network uses the weights and biases which are randomely initialised and with the differentiable non-linear functions it helps predict the output variable referred as y-hat ($\hat{y}$) or y-predicted ($y_{pred}$). This predicted value is improved over the epochs(re-iteration) by calcularing the error loss, a mathematical way to understand by how much predicted value is deviated from the actual output value. There are 3 different layers in the neural networks Input Layer, Hidden Layer and Output Layer. Refer the diagram shown above. 

In short, there are below 4 steps involved in order to make neural networks learn:

1. Initialize weights(W) and bias(b) just once in the whole cycle (epochs),
2. Feed Forward in the network - basically weighted sum of inputs plus bias is fed to non-linear function (widely used sigmoid) at neuron level and this operation is carried out layer on layer. 
3. Error Loss - differnt error loss methods used are 'Squared Error Loss', 'RMSE (Root Mean Sqared Error Loss)' and 'Cross Entropy'.
4. Backpropagation (Gradient Descent) - this is basically used to modify the weights and bias so that it helps in reducing the error loss. (Inspired from Taylor's Series). 

In the diagram, the circle shown in blue as well as in green is called neurons and its get activated when weighted sum of inputs are fed to it followed by non-linear function (like sigmoid). There are two main events which takes place at the neuron 1. pre-activation 2. activation

1. Pre-activation (denoted by a) - This aggregates the weighted sum of inputs plus bias and fed as an input to the neuron.
2. Activation (denoted by h) - Here a non-linear function like sigmoid function is applied on the  pre-activation function output and then the activation output is passed to the subsequent nerons untill it reaches to the output neurons.

Naming convention:

Blue Neurons are called Hidden Layer which are above input layer (x) and below the output layer which is green color neuron. We can have many layers depending upon the compexity and accuracy needed to learn the model. 

x = input

Function:

a = pre-activation function, W = weight, b = bias

Output:  $y_{pred} = h_{21}$

h = activation function (output of the hidden layers)

For the simplicity of the terminoligies I have denoted the output layer with same convention as hidden layer but have colored it green to differentiate.
$$
W_1 = 
\begin{bmatrix} 
w_{111} & w_{112} \\
w_{211} & w_{212}\\
\end{bmatrix}
\quad
, 
b_1 = 
\begin{bmatrix} 
b_{11}  \\
b_{12}
\end{bmatrix}
\quad
,
W_2 = 
\begin{bmatrix} 
w_{121} & w_{221} \\
\end{bmatrix}
\quad
,
b_2 = b_{21}
$$

$$
X = 
\begin{bmatrix} 
x_{11} & x_{12} \\
x_{21} & x_{22}\\
x_{31} & x_{32}\\
{..} & {..}\\
x_{n1} & x_{n2}\\
\end{bmatrix}
\quad
$$

While the model is fit, input is fed to the model row by row until all the row is exhausted (and we denote this one complete cyle as epochs ), hence the first row of the X is shown below:
$$
X_{1} = \begin{bmatrix} x_{11} \\ 
x_{12} \\\end{bmatrix}\quad
$$

Therefore, $a_{11}, a_{12}$, is calculated by Matrix multiplication of $W_1, X_1, b1$ as shown below:
$$
\begin{bmatrix} 
a_{11}  \\
a_{12}
\end{bmatrix}
\quad
=
W_1 * X_1 + b1, \\ 
i.e.
\begin{bmatrix} 
a_{11}  \\
a_{12}
\end{bmatrix}
\quad
=
\begin{bmatrix} 
w_{111} & w_{112} \\
w_{211} & w_{212}\\
\end{bmatrix}
\quad
*
 \begin{bmatrix} x_{11} \\ 
x_{12} \\\end{bmatrix}\quad
+
\begin{bmatrix} 
b_{11}  \\
b_{12}
\end{bmatrix}
\quad
\\
= 
\begin{bmatrix} 
w_{111}.x_{11} + w_{112}.x_{12} \\
w_{211}.x_{11} + w_{212}.x_{12} 
\end{bmatrix}
\quad
+
\begin{bmatrix} 
b_{11}  \\
b_{12}
\end{bmatrix}
\quad
\\
$$
Replacing the values with randomly initialised weigths and bias.

Lets initialise the weights, and bias:
$$
W_1 = 
\begin{bmatrix} 
0.1 & 0.2 \\
-0.3 & 0.4 \\
\end{bmatrix}
\quad
,
b_1 = 
\begin{bmatrix} 
0  \\
0
\end{bmatrix}
\quad
,
W_2 = 
\begin{bmatrix} 
0.5 & 0.6 \\
\end{bmatrix}
\quad
,
b_2 = 0
$$
Assume X to be:
$$
X = 
\begin{bmatrix} 
0.91 & 9.17 \\
0.85 & 3.86\\
-2.39 & 7.39\\
{..} & {..}\\
2.50 & 5.77\\
\end{bmatrix}
\quad
$$
First row of the input is :
$$
X_{1} = \begin{bmatrix} 0.91 \\ 
9.17 \\\end{bmatrix}\quad
$$


Lets calculate $a_{11} and \ a_{12}$,
$$
a_{11} = [(0.1 * 0.91) + (0.2 * 9.17)] + 0 = 1.92 \\
a_{12} = [(-0.3 * 0.91) + (0.4 * 9.17)] + 0 = 3.39
$$
Lets calculate $h_{11}, \ h_{12}$,
$$
h_{11} \ = \ \frac{1}{1+ e ^{-a_{11}}} = \ \frac{1}{1+ e ^{-1.92}} = 0.872
$$
Similary, 
$$
h_{12} \ = \ \frac{1}{1+ e ^{-a_{12}}} = \ \frac{1}{1+ e ^{-3.39}} = 0.967
$$

Lets calculate $a_{21}, \ h_{21}$, 		
$$
a_{21} = [(0.5 * 0.872) + (0.6 * 0.967)] + 0 = 1.01 \\
h_{21} \ = \ \frac{1}{1+ e ^{-a_{21}}} = \ \frac{1}{1+ e ^{-1.01}} = 0.733
$$


<img src="/Users/anupam7936/afb/Blogs/image-Neural1.png" alt="image-20200414070040022" style="zoom:50%;" />



Here, output $y_{pred}$ is same as $h_{21}$, because the output is binary hence sigmoid function suffice the requirement. Had it been a multi class classification problem then we would have used other non-linear function like softmax function to predict the output.
$$
y_{pred} = softmax(a_{21})
$$
Congratulations!, we have successfully completed the feed forward understanding of the network. Now the other two crucial part to be addressed are "Error Loss" and "Backpropagation", we will see both in next section. 

​						