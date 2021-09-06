import numpy as np 
import utility as ut
import os
import tensorflow as tf
import plot
import matplotlib.pyplot as plt
import time


class DataGen:

    def __init__(self, dim, gen_path, folder):
        self.gen_path = gen_path 
        self.dim = dim
        if not os.path.isdir(folder):
            os.mkdir(folder)
        self.folder = folder

    @ut.timer
    def random_on_atr(self, length, burn_in):
        x0 = np.random.uniform(size=self.dim)
        x0 = self.gen_path(x0, burn_in)[-1]
        return self.gen_path(x0, length)

    @ut.timer
    def create_random_dataset(self, num_paths, length, burn_in, name):
        x_data = np.zeros((num_paths * length, self.dim))
        y_data = np.zeros((num_paths * length, self.dim))
        i, j = 0, length
        for path_id in range(num_paths):
            print('working on path #{}'.format(path_id), end='\r')
            path = self.random_on_atr(length+1, burn_in)
            x_data[i: j] = path[:length]
            y_data[i: j] = path[1:length+1]
            i += length
            j += length
        x_file = '{}/{}_x.npy'.format(self.folder, name) 
        y_file = '{}/{}_y.npy'.format(self.folder, name) 
        np.save(x_file, x_data)
        np.save(y_file, y_data)



class LSTMForgetBlock(tf.keras.layers.Layer):
    def __init__(self, num_nodes, dtype=tf.float64):
        super().__init__(name='LSTMForgetBlock', dtype=dtype)
        self.W_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_f', use_bias=False)
        self.U_f = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_f')
        self.W_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_i', use_bias=False)
        self.U_i = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_i')
        self.W_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_o', use_bias=False)
        self.U_o = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_o')
        self.W_c = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='W_c', use_bias=False)
        self.U_c = tf.keras.layers.Dense(num_nodes, dtype=dtype, name='U_c')

    def call(self, x, h, c):
        f = tf.keras.activations.sigmoid(self.W_f(x) + self.U_f(h))
        i = tf.keras.activations.sigmoid(self.W_i(x) + self.U_i(h))
        o = tf.keras.activations.sigmoid(self.W_o(x) + self.U_o(h))
        c_ = tf.keras.activations.tanh(self.W_c(x) + self.U_c(h))
        c = f*c + i*c_
        return o*tf.keras.activations.tanh(c), c


class GeoMap(tf.keras.models.Model):
     
    def __init__(self, dim, num_nodes, num_layers, folder, name='GeoMap', dtype=tf.float32):
        super().__init__(name=name, dtype=dtype)
        self.dim = dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers
        folder += '/' + name
        if not os.path.isdir(folder):
            os.mkdir(folder)
        self.folder = folder
        self.ls = [tf.keras.layers.Dense(units=num_nodes, activation=tf.keras.activations.tanh, dtype=dtype) for _ in range(num_layers)]
        self.final_dense = tf.keras.layers.Dense(units=dim, activation=None, dtype=dtype)


    def call(self, x):
        #x = tf.concat(args, axis=1)
        for i in range(self.num_layers):
            x = self.ls[i](x)
        y = self.final_dense(x)
        return y 

    def learn(self, x_data, y_data, epochs, learning_rate=1e-3):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                loss = tf.reduce_mean(tf.reduce_sum((self.call(x_data) - y_data)**2, axis=1))
                print('epoch #{}: squared loss = {}'.format(epoch + 1, loss), end='\r')
            grads = tape.gradient(loss, self.trainable_weights)
            optimizer.apply_gradients(zip(grads, self.trainable_weights))

    def save(self, model_id=''):
        super().save_weights(self.folder + '/weights_' + str(model_id))
    
    def load(self, model_id):
        weight_file = self.folder + '/weights_' + str(model_id)
        if os.path.isfile(weight_file + '.index'):
            super().load_weights(weight_file).expect_partial()
        else:
            print('Weight file does not exist for model id = {}. Weights were not loaded.'.format(model_id))
 
    def gen_path(self, x0, length):
        path = np.zeros((length, self.dim))
        path[0] = x0
        for i in range(1, length):
            path[i] = self.call(path[np.newaxis, i-1])[0]
        return path
    
    def visualize(self, path, plot_type='3d', coords=[0, 1, 2]):
        learned_path = self.gen_path(path[0], len(path))
        timeline = np.arange(len(path))
        colors = ['#04471C', '#EA3788', '#2D0320']
        #print(learned_path)
        if plot_type == '3d':
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            ax.plot3D(path[:, coords[0]], path[:, coords[1]], path[:, coords[2]], label='true', c=colors[0])
            ax.plot3D(learned_path[:, coords[0]], learned_path[:, coords[1]], learned_path[:, coords[2]], label='learned', c=colors[1])
            ax.scatter3D(path[0, coords[0]], path[0, coords[1]], path[0, coords[2]], s=50, c=colors[2], label='x0')
            plt.legend()
            plt.show()

        elif plot_type == '1d':
            sp = plot.SignalPlotter(signals=[path, learned_path])
            sp.plot_signals(labels=['true', 'learned'], colors=colors, coords_to_plot=coords,\
                            styles = [{'linestyle':'solid'}, {'linestyle':'dashed'}], show=True)






class GeoMapLSTM(GeoMap):
     
    def __init__(self, dim, num_nodes, num_layers, folder, name='GeoMapLSTM', dtype=tf.float32):
        super().__init__(dim, num_nodes, num_layers, folder, name, dtype)
        self.ls = [LSTMForgetBlock(num_nodes, dtype=dtype) for _ in range(num_layers)]

    def call(self, x):
        h = tf.zeros_like(x)
        c = tf.zeros((x.shape[0], self.num_nodes), dtype=self.dtype)
        for i in range(self.num_layers):
            h, c = self.ls[i](x, h, c)
            #h = self.batch_norm(h)
            #c = self.batch_norm(c)
        y = self.final_dense(h)
        return y


class SpeedCompare:
    def __init__(self, func_1, func_2):
        self.f1 = func_1
        self.f2 = func_2

    def test(self, num_iters, *args):
        f1_start = time.time()
        for _ in range(num_iters):
            self.f1(*args)
        f1_end = time.time()
        f2_start = time.time()
        for _ in range(num_iters):
            self.f2(*args)
        f2_end = time.time()
        f1_avg = (f1_end - f1_start) / num_iters
        f2_avg = (f2_end - f2_start) / num_iters
        print('average time required for {} is {}'.format(self.f1.__name__, f1_avg))
        print('average time required for {} is {}'.format(self.f2.__name__, f2_avg))