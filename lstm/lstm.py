import tensorflow as tf
import numpy as np

class TensorFlowLSTM():
    def __init__(self, hidden_size, window_size, observation_dimension, num_layers=1):
        self.num_layers=num_layers
        self.window_size = window_size
        self.hidden_size = hidden_size
        self.observation_dimension = observation_dimension

        self.model = self.build_model()

    def __call__(self):
        return self.model
    
    def __call__(self, input):
        return self.model(input)

    def build_model(self, kernel_seed=123, recurrent_seed=123):

        kernel_init = tf.keras.initializers.GlorotUniform(seed=kernel_seed)
        recurrent_init = tf.keras.initializers.Orthogonal(seed=recurrent_seed)

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(
                self.hidden_size,
                activation="tanh",
                name="LSTM_1",
                return_sequences=True,
                kernel_initializer=kernel_init,
                recurrent_initializer=recurrent_init,
            ), 
            tf.keras.layers.Dense(self.observation_dimension, name="DenseLayer")]
        )
        model.compile(optimizer=tf.keras.optimizers.Adam(), metrics=["mse"], loss=tf.keras.losses.MeanSquaredError())
        return model

    def lstm_step(self, u_t, h, c):
        """Executes one LSTM step for the Lyapunov exponent computation

        Args:
            u_t (tf.EagerTensor): differential equation at time t
            h (tf.EagerTensor): LSTM hidden state at time t
            c (tf.EagerTensor): LSTM cell state at time t
            model (keras.Sequential): trained LSTM
            idx (int): index of current iteration
            dim (int, optional): dimension of the lorenz system. Defaults to 3.

        Returns:
            u_t (tf.EagerTensor): LSTM prediction at time t/t+1
            h (tf.EagerTensor): LSTM hidden state at time t+1
            c (tf.EagerTensor): LSTM cell state at time t+1
        """
        if type(u_t) is np.ndarray or np.array:
            u_t = tf.convert_to_tensor(u_t)
        z = tf.keras.backend.dot(u_t, self.model.layers[0].cell.kernel)
        z += tf.keras.backend.dot(h, self.model.layers[0].cell.recurrent_kernel)
        z = tf.keras.backend.bias_add(z, self.model.layers[0].cell.bias)

        z0, z1, z2, z3 = tf.split(z, 4, axis=1)

        i = tf.sigmoid(z0)
        f = tf.sigmoid(z1)
        c_new = f * c + i * tf.tanh(z2)
        o = tf.sigmoid(z3)

        h_new = o * tf.tanh(c_new)

        u_t = tf.reshape(
            tf.matmul(h_new, self.model.layers[1].get_weights()[0])
            + self.model.layers[1].get_weights()[1],
            shape=(1, self.observation_dimension),
        )
        return u_t, h_new, c_new


    def closed_loop_prediction(self, x, prediction_length):

        if type(x) is np.ndarray or np.array:
            x = tf.convert_to_tensor(x)
        u_t = x[:, 0, :]
        h = tf.Variable(self.model.layers[0].get_initial_state(x)[0], trainable=False)
        c = tf.Variable(self.model.layers[0].get_initial_state(x)[1], trainable=False)
        pred = np.zeros(shape=(prediction_length, self.observation_dimension))
        pred[0, :] = u_t

        # prepare h,c and c from first window
        for i in range(1, self.window_size + 1):
            u_t = x[:, i - 1, :]
            u_t, h, c = self.lstm_step(u_t, h, c)
            pred[i, :] = u_t
        for i in range(self.window_size + 1, prediction_length):
            u_t, h, c = self.lstm_step(u_t, h, c)
            pred[i, :] = u_t

        return pred