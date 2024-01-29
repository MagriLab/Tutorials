# Tutorials: Predictions in Chaotic Dynamical Systems 

This is a tutorial to employ echo state networks (ESNs) and long short-term memory networks (LSTMs) for the prediction and analysis of chaotic dynamics. 
This library contains both `Tensorflow` and `PyTorch` implementations for the LSTM and employs the [Magrilab/EchoStateNetwork](https://github.com/MagriLab/EchoStateNetwork). Please note that encountered issues may be addressed there. 

The example system found here is the Lorenz 63 system, which is found in `dynamicalsystems.equations`

$$
\begin{aligned}
		&\dfrac{\mathrm{d}x}{\mathrm{d}t} = \sigma (y-x) \\
		&\dfrac{\mathrm{d}y}{\mathrm{d}t} = x (\rho-z) - y \\
		&\dfrac{\mathrm{d}z}{\mathrm{d}t} = xy - \beta z.
\end{aligned}
$$

## **Tutorials: LSTM and ESN to learn Lorenz-63**
The tutorial for the LSTM can be found in `LSTM_Tutorial_Lorenz63.ipynb` and the ESN can be found in  `ESB_Tutorial_Lorenz63.ipynb`.


## **Example: Attractor reconstruction by reference (black), LSTM (blue) and ESN (red):**
<p align='center'>
<img src="media/network_attractor.png"/>
</p>

## **Requirements:**
You can find a list of requirements in `requirements.txt`. We recommend installing the requirements in a conda environment. 

For numpy version > 1.15, there may be a np.int error occurring; this is due to a missing bugfix from skopt. Follow the instructions of this issue:[Resolve Deprecated Numpy Attribute Error np.int]([https://github.com/MagriLab/EchoStateNetwork](https://github.com/MagriLab/Tutorials/issues/1)https://github.com/MagriLab/Tutorials/issues/1)


