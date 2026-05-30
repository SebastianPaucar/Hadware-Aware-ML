import tensorflow as tf
K = tf.keras.backend

class _tcvae():
    def __init__(self, gamma=6.0, alpha=1.0, beta=1.0, dataset_size=None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.dataset_size = dataset_size

    def __call__(self, mu, log_var):
        mu      = tf.cast(mu,      dtype='float32')
        log_var = tf.cast(log_var, dtype='float32')
        mi, tc, dw_kl = self._decompose(mu, log_var)
        return self.alpha * mi + self.gamma * tc + self.beta * dw_kl

    def _decompose(self, z_mean, z_log_var):
        eps = tf.random.normal(tf.shape(z_mean), dtype='float32')
        z   = z_mean + tf.exp(0.5 * z_log_var) * eps

        M = tf.cast(tf.shape(z)[0], tf.float32)
        N = tf.cast(self.dataset_size if self.dataset_size is not None else tf.shape(z)[0], tf.float32)

        z_i       = tf.expand_dims(z,         axis=1)
        mu_j      = tf.expand_dims(z_mean,    axis=0)
        log_var_j = tf.expand_dims(z_log_var, axis=0)

        log_q_z_given_x = self._log_normal(z_i, mu_j, log_var_j)        # (M, M, D)

        log_q_z_given_x_sum_d = tf.reduce_sum(log_q_z_given_x, axis=2)  # (M, M)
        log_q_z = tf.reduce_logsumexp(log_q_z_given_x_sum_d, axis=1) - tf.math.log(M * N)

        log_q_z_given_xi = tf.reduce_sum(self._log_normal(z, z_mean, z_log_var), axis=1)

        log_q_z_product = (
            tf.reduce_sum(tf.reduce_logsumexp(log_q_z_given_x, axis=1), axis=1)
            - tf.math.log(M * N)
        )

        log_p_z = tf.reduce_sum(self._log_standard_normal(z), axis=1)

        mi    = tf.reduce_mean(log_q_z_given_xi - log_q_z)
        tc    = tf.reduce_mean(log_q_z           - log_q_z_product)
        dw_kl = tf.reduce_mean(log_q_z_product   - log_p_z)

        return mi, tc, dw_kl

    @staticmethod
    def _log_normal(z, mu, log_var):
        return -0.5 * (tf.math.log(2.0 * 3.141592653589793) + log_var + tf.square(z - mu) / tf.exp(log_var))

    @staticmethod
    def _log_standard_normal(z):
        return -0.5 * (tf.math.log(2.0 * 3.141592653589793) + tf.square(z))
