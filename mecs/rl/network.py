import tensorflow as tf

class Estimator():
    def __init__(self, state_size, output_size, action_size):
        self.state_size = state_size
        self.output_size = output_size
        self.action_size = action_size
        self.learning_rate = 1e-4

    def build_network(self):
        self.state = tf.placeholder(shape=[None, self.state_size], dtype=tf.float32, name='state')
        self.target_reward = tf.placeholder(shape=[None], dtype=tf.float32, name='target_reward')
        self.actions = tf.placeholder(shape=[None, self.action_size], dtype=tf.float32, name='actions')
        batch_size = tf.shape(self.state)[0]

        '''
        뉴럴넷 구성
        '''
        # self.action_values = 네트워크 아웃풋
        # self.predictions = tf.nn.softmax(self.action_values)

    def predict():
        pass

    def update():
        pass
