import tensorflow as tf
import tensorflow.contrib.layers as tflayers
import tensorflow.contrib.rnn as RNN
import numpy as np
import pandas as pd
from lazyloading import define_scope
from tensorboard import batch_index

def rnn_data(data, time_steps, labels=False):
    """
    creates new data frame based on previous observation
      * example:
        l = [1, 2, 3, 4, 5]
        time_steps = 2
        -> labels == False [[1, 2], [2, 3], [3, 4]]
        -> labels == True [3, 4, 5]
    """
    rnn_df = []
    for i in range(len(data) - time_steps):
        if labels:
            try:
                rnn_df.append(data.iloc[i + time_steps].as_matrix())
            except AttributeError:
                rnn_df.append(data.iloc[i + time_steps])
        else:
            data_ = data.iloc[i: i + time_steps].as_matrix()
            rnn_df.append(data_ if len(data_.shape) > 1 else [[i] for i in data_])

    return np.array(rnn_df, dtype=np.float32)

def split_data(data, val_size=0.1, test_size=0.1):
    """
    splits data to training, validation and testing parts
    """
    ntest = int(round(len(data) * (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest]) * (1 - val_size)))

    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest], data.iloc[ntest:]

    return df_train, df_val, df_test

def prepare_data(data, time_steps, labels=False, val_size=0.1, test_size=0.1):
    """
    Given the number of `time_steps` and some data,
    prepares training, validation and test data for an lstm cell.
    """
    df_train, df_val, df_test = split_data(data, val_size, test_size)
    return (rnn_data(df_train, time_steps, labels=labels),
            rnn_data(df_val, time_steps, labels=labels),
            rnn_data(df_test, time_steps, labels=labels))

def generate_data(fct, x, time_steps, seperate=False):
    """generates data with based on a function fct"""
    data = fct(x)
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    train_x, val_x, test_x = prepare_data(data['a'] if seperate else data, time_steps)
    train_y, val_y, test_y = prepare_data(data['b'] if seperate else data, time_steps, labels=True)
    return dict(train=train_x, val=val_x, test=test_x), dict(train=train_y, val=val_y, test=test_y)


class Config(object):
    hidden_size = 10
    time_steps = 5
    input_size = 1
    rnn_layers = [{"hidden_size": hidden_size}, {'hidden_size': hidden_size, 'keep_prob': 0.5}]
    dense_layers = [2]
    lr = 0.1
    lr_decay = 0.8
    max_epoch = 1000
    max_max_epoch = 2000
  
  
class lstmRegressionModel:
    
    def __init__(self, data, label,statec1,statec2,stateh1,stateh2 ,config):
        self.data = data
        self.label = label
        self._config = config
        self.statec1=statec1
        self.statec2=statec2
        self.stateh1=stateh1
        self.stateh2=stateh2
        
        self.prediction
        self.loss
        self.optimize
        self.error

    @define_scope
    def prediction(self):
        def lstm_cells(layers):
            if isinstance(layers[0], dict):
                #BasicLSTMCell的第一个参数指定LSTM单元中cell(以及输出)的大小（hidden_size）
                #rnn_layers = [{"hidden_size": hidden_size}, {'hidden_size': hidden_size, 'keep_prob': 0.5}]
                return [RNN.DropoutWrapper(RNN.BasicLSTMCell(layer['hidden_size'],
                                                                                   state_is_tuple=True),
                                                      layer['keep_prob'])
                        if layer.get('keep_prob') else RNN.BasicLSTMCell(layer['hidden_size'],
                                                                                    state_is_tuple=True)
                        for layer in layers]
            return [RNN.BasicLSTMCell(hs, state_is_tuple=True) for hs in layers]
    
        def dnn_layers(input, dense_layers):
            #dense_layers=[2]
            if dense_layers and isinstance(dense_layers, dict):
                return tflayers.stack(input, tflayers.fully_connected,
                                      dense_layers['layers'],
                                      activation=dense_layers.get('activation'),
                                      dropout=dense_layers.get('dropout'))
            elif dense_layers:
                return tflayers.stack(input, tflayers.fully_connected, dense_layers)
            else:
                return input
            
     
        stacked_lstm = RNN.MultiRNNCell(lstm_cells(self._config.rnn_layers), state_is_tuple=True)
        
        LSTMStateTuple1=RNN.LSTMStateTuple(self.statec1,self.stateh1)
        LSTMStateTuple2=RNN.LSTMStateTuple(self.statec2,self.stateh2)
        state=(LSTMStateTuple1,LSTMStateTuple2)
        
        output, states = tf.nn.dynamic_rnn(stacked_lstm, self.data,initial_state=state, dtype=tf.float32,)

        output = output[:,-1,:]
        output = dnn_layers(output, self._config.dense_layers)
        return tflayers.linear(output, 1),states
    
    
    @define_scope
    def loss(self):
        h=self.prediction[0]
        return tf.losses.mean_squared_error(self.label, h)
        
    @define_scope
    def error(self):
        return tf.sqrt(self.loss, name='sqrt_of_loss')

    @define_scope
    def optimize(self):
        self._lr = tf.Variable(self._config.lr, name='learning_rate', trainable=False)
        return tf.train.AdamOptimizer(self._lr).minimize(self.loss)

    def update_lr(self, sess, nepoch):
        lr_decay = self._config.lr_decay ** (nepoch/self._config.max_epoch)#max(nepoch + 1 - self._config.max_epoch, 0.0)
        sess.run(tf.assign(self._lr, self._config.lr*lr_decay))
def batch_index(data,time_step):
    batch_index1=[]
    for i in range(int(len(data)/time_step)+1):
        batch_index1.append(time_step*i)
    return batch_index1  
  

def main():
   
    config = Config()
    tf.logging.set_verbosity(tf.logging.INFO)
    t = np.linspace(0, 10, 100)
    X, y = generate_data(np.sin, t, config.time_steps, seperate=False)
    # X['train'] : bachsize*timesteps*1
    # y['train'] : batchsize*1
   
    train = tf.placeholder(tf.float32, [config.time_steps, config.time_steps, config.input_size], name='X')
    label = tf.placeholder(tf.float32, [config.time_steps, 1], name='Y')
    state_c1 = tf.placeholder(tf.float32, [config.time_steps,config.hidden_size],name='c1')
    state_c2 = tf.placeholder(tf.float32, [config.time_steps,config.hidden_size],name='c2')
    state_h1 = tf.placeholder(tf.float32, [config.time_steps,config.hidden_size],name='h1')
    state_h2 = tf.placeholder(tf.float32, [config.time_steps,config.hidden_size],name='h2')
    
    model = lstmRegressionModel(train, label, state_c1,state_c2,state_h1,state_h2,config)     
    
    zeros=np.zeros([config.time_steps,config.hidden_size],dtype=np.float32)
    a=batch_index(X['train'],config.time_steps)
    with tf.Session() as sess:
        writer =  tf.summary.FileWriter('./ops_logs/lstmPredict', sess.graph)
        sess.run(tf.global_variables_initializer())
        for _ in range(config.max_max_epoch):
           
            for i in range(len(a)-1):
                p=[]
                if(i==0):
                    prediction,state=sess.run(model.prediction,{train:X['train'][a[i]:a[i+1]],label:y['train'][a[i]:a[i+1]],state_c1:zeros,state_c2:zeros,state_h1:zeros,state_h2:zeros})
                    pre_state=state
                    p.append(prediction)
                elif(i<(len(a)-2)):
                    prediction,state=sess.run(model.prediction,{train:X['train'][a[i]:a[i+1]],label:y['train'][a[i]:a[i+1]],state_c1:pre_state[0].c,state_c2:pre_state[1].c,state_h1:pre_state[0].h,state_h2:pre_state[1].h})
                    pre_state=state
                    p.append(prediction)
                else:
                    prediction,state=sess.run(model.prediction,{train:X['train'][a[i]:a[i+1]],label:y['train'][a[i]:a[i+1]],state_c1:pre_state[0].c,state_c2:pre_state[1].c,state_h1:pre_state[0].h,state_h2:pre_state[1].h})
                    #error_=sess.run(model.error,{train:X['train'][a[i]:a[i+1]],label:y['train'][a[i]:a[i+1]],state_c1:pre_state[0].c,state_c2:pre_state[1].c,state_h1:pre_state[0].h,state_h2:pre_state[1].h})
                    p.append(prediction)
            print(p)
            #if(_%100==0):
                
        writer.close()
  
    
                    
if __name__ == '__main__':
    main()        