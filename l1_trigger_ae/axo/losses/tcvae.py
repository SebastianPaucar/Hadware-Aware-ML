import tensorflow as tf
K = tf.keras.backend

class _tcvae():
    def __init__(self, gamma=6.0, alpha=1.0, beta=1.0, scale_factor=1, dataset_size=None):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = float(alpha)
        self.beta  = float(beta)
        self.scale_factor = float(scale_factor)
        self.dataset_size = dataset_size

    def __call__(self, z, mu, log_var):
        z       = tf.cast(z,       dtype='float32')
        mu      = tf.cast(mu,      dtype='float32')
        log_var = tf.cast(log_var, dtype='float32')
        mi, tc, dw_kl = self._decompose(z, mu, log_var)
        raw_loss = self.alpha * mi + self.gamma * tc + self.beta * dw_kl
        return raw_loss * self.scale_factor

    def _decompose(self, z, z_mean, z_log_var):
        M = tf.cast(tf.shape(z)[0], tf.float32)
        N = tf.cast(self.dataset_size, tf.float32)
        log_weight = tf.math.log(M)
        
        z_i       = tf.expand_dims(z,         axis=1)
        mu_j      = tf.expand_dims(z_mean,    axis=0)
        log_var_j = tf.expand_dims(z_log_var, axis=0)

        log_q_z_given_x = self._log_normal(z_i, mu_j, log_var_j)        # (M, M, D)
        
        # log q(z)
        log_q_z_given_x_sum_d = tf.reduce_sum(log_q_z_given_x, axis=2)  # (M, M)
        log_q_z = tf.reduce_logsumexp(log_q_z_given_x_sum_d, axis=1) - log_weight

        # log q(z|x)
        log_q_z_given_xi = tf.reduce_sum(self._log_normal(z, z_mean, z_log_var), axis=1)
        
        # Apply log_weight subtraction BEFORE summing across latent dimensions
        log_q_z_per_dim = tf.reduce_logsumexp(log_q_z_given_x, axis=1) - log_weight # (M, D)
        log_q_z_product = tf.reduce_sum(log_q_z_per_dim, axis=1)                    # (M,)
  
        # Extract D (latent dimensions)
        D = tf.cast(tf.shape(z)[-1], tf.float32)

        # log p(z)
        log_p_z = tf.reduce_sum(self._log_standard_normal(z), axis=1)

        mi    = tf.reduce_mean(log_q_z_given_xi - log_q_z) / D 
        tc    = tf.reduce_mean(log_q_z           - log_q_z_product) / D
        dw_kl = tf.reduce_mean(log_q_z_product   - log_p_z)  / D

        return mi, tc, dw_kl

    @staticmethod
    def _log_normal(z, mu, log_var):
        log_var_safe = tf.clip_by_value(log_var, -10.0, 10.0)
        return -0.5 * (tf.math.log(2.0 * 3.141592653589793) + log_var_safe + tf.square(z - mu) * tf.exp(-log_var_safe))

    @staticmethod
    def _log_standard_normal(z):
        return -0.5 * (tf.math.log(2.0 * 3.141592653589793) + tf.square(z))
