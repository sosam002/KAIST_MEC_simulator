import tensorflow as tf

from rllab.core.serializable import Serializable

from sac.misc.mlp import MLPFunction, MLPFunctionWithAlpha
from sac.misc import tf_utils

class NNVFunction(MLPFunction):

    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), name='vf'):
        Serializable.quick_init(self, locals())

        self._Do = env_spec.observation_space.flat_dim
        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        super(NNVFunction, self).__init__(
            name, (self._obs_pl,), hidden_layer_sizes)


class NNQFunction(MLPFunction):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), name='qf'):
        Serializable.quick_init(self, locals())

        self._Da = env_spec.action_space.flat_dim
        self._Do = env_spec.observation_space.flat_dim

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        super(NNQFunction, self).__init__(
            name, (self._obs_pl, self._action_pl), hidden_layer_sizes)

class NNRFunction(MLPFunctionWithAlpha):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), name='rf', with_alpha = True, const_alpha=False):
        Serializable.quick_init(self, locals())

        self._Da = env_spec.action_space.flat_dim
        self._Do = env_spec.observation_space.flat_dim

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )

        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        super(NNRFunction, self).__init__(
            name, (self._obs_pl, self._action_pl), (self._obs_pl,), hidden_layer_sizes, output_nonlinearity=tf.nn.sigmoid, with_alpha=with_alpha, const_alpha=const_alpha)


class NNDiscriminatorFunction(MLPFunction):
    def __init__(self, env_spec, hidden_layer_sizes=(100, 100), num_skills=None):
        assert num_skills is not None
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)

        self._Da = env_spec.action_space.flat_dim
        self._Do = env_spec.observation_space.flat_dim

        self._obs_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Do],
            name='observation',
        )
        self._action_pl = tf.placeholder(
            tf.float32,
            shape=[None, self._Da],
            name='actions',
        )

        self._name = 'discriminator'
        self._input_pls = (self._obs_pl, self._action_pl)
        self._layer_sizes = list(hidden_layer_sizes) + [num_skills]
        self._output_t = self.get_output_for(*self._input_pls)

class BetaFunction(Serializable):

    def __init__(self, name):
        super(BetaFunction, self).__init__()
        Serializable.quick_init(self, locals())
        self._log_beta_t = self.get_beta_for()

    def get_beta_for(self, reuse=False):
        with tf.variable_scope('beta', reuse=reuse):
            log_beta_t = tf.get_variable('log_beta', dtype=tf.float32, initializer=0.0)
        return log_beta_t

    def get_params_beta(self, **tags):
        if len(tags) > 0:
            raise NotImplementedError

        scope = tf.get_variable_scope().name
        scope += '/' + 'beta' + '/' if len(scope) else 'beta' + '/'

        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope
        )


class AlphaFunction(Serializable):

    def __init__(self, name):
        super(AlphaFunction, self).__init__()
        Serializable.quick_init(self, locals())
        self._alpha_t = self.get_alpha_for()

    def get_alpha_for(self, reuse=False):
        regularizer = tf.contrib.layers.l2_regularizer(0.001)
        with tf.variable_scope('alpha', reuse=reuse):
            alpha_t = 0.49 * tf.nn.sigmoid(tf.get_variable('alpha', dtype=tf.float32, initializer=0.0, regularizer=regularizer)) + 0.5
        return alpha_t

    def get_params_alpha(self, **tags):
        if len(tags) > 0:
            raise NotImplementedError

        scope = tf.get_variable_scope().name
        scope += '/' + 'alpha' + '/' if len(scope) else 'alpha' + '/'

        return tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope
        )
