import tensorflow as tf, pandas as pd
from tensorflow.contrib.nn import alpha_dropout

from . import metadata
from .utils import utils

class Model(object):
    """DNN model for kkbox music recommendation"""
    logger = utils.logger(__name__)

    def __init__(self, model_dir):
        """

        :param model_dir: Model checkpoint directory path
        """
        self.model_dir = model_dir
        self.mapper = None
        self.share_emb = True
        pass

    def base_features(self, p, features, label, mode):
        """Handle graph about all kind of variables initialization, weighted sum of all features and relevant
          weights, concat features of members, songs

        :param features: Dictionary object include all input feature
        :param label: Reserve for refactor
        :param mode: Reserve for refactor
        :return: None
        """
        with tf.variable_scope("init", reuse=tf.AUTO_REUSE) as scope:
            uniform_init_fn = tf.glorot_uniform_initializer()
            # self.b_global = tf.Variable(uniform_init_fn(shape=[]), name="b_global")
            # Embedding init
            with tf.variable_scope("embedding"):
                self.emb = {}

                emb_cols = metadata.EMB_COLS.copy()
                # Edit embedding size to receive the hyper parameter setting
                for key in ('song_id', 'artist_name', 'composer', 'lyricist'):
                    emb_cols[key] = p.embedding_size

                for colname, dim in emb_cols.items():
                    n_unique = len(self.mapper[colname].classes_)
                    self.emb[colname] = self.get_embedding_var(
                        [n_unique, dim], uniform_init_fn, 'emb_{}'.format(colname))

                if not self.share_emb:
                    self.logger.info('Split embedding of song_query and song_id!')
                    # For NeuMFModel, try to use different embedding vocabularies
                    song_query_len = len(self.mapper['song_id'].classes_)
                    song_query_emb_dim = emb_cols['song_id']
                    self.emb['song_query'] = self.get_embedding_var(
                        [song_query_len, song_query_emb_dim],
                        uniform_init_fn,
                        'emb_song_query'
                    )

        with tf.variable_scope("members") as scope:
            self.city = tf.nn.embedding_lookup(self.emb['city'], features['city'])
            self.gender = tf.nn.embedding_lookup(self.emb['gender'], features['gender'])
            self.registered_via = tf.nn.embedding_lookup(self.emb['registered_via'], features['registered_via'])
            self.registration_init_time = features['registration_init_time'][:, tf.newaxis]
            self.expiration_date = features['expiration_date'][:, tf.newaxis]
            self.msno_age_catg = tf.nn.embedding_lookup(self.emb['msno_age_catg'], features['msno_age_catg'])
            self.msno_age_num = features['msno_age_num'][:, tf.newaxis]
            self.msno_tenure = features['msno_tenure'][:, tf.newaxis]

            self.weighted_sum(features, 'song_id', 'msno_pos_query_hist', ['msno_pos_query_count'])
            self.weighted_sum(features, 'song_id', 'msno_neg_query_hist', ['msno_neg_query_count'])
            self.weighted_sum(features, 'artist_name', 'msno_artist_name_hist', ['msno_artist_name_count', 'msno_artist_name_mean'])
            self.weighted_sum(features, 'composer', 'msno_composer_hist', ['msno_composer_count', 'msno_composer_mean'])
            self.weighted_sum(features, 'genre_ids', 'msno_genre_ids_hist', ['msno_genre_ids_count', 'msno_genre_ids_mean'])
            self.weighted_sum(features, 'language', 'msno_language_hist', ['msno_language_count', 'msno_language_mean'])
            self.weighted_sum(features, 'lyricist', 'msno_lyricist_hist', ['msno_lyricist_count', 'msno_lyricist_mean'])
            self.weighted_sum(features, 'source_screen_name', 'msno_source_screen_name_hist', ['msno_source_screen_name_count'])
            self.weighted_sum(features, 'source_system_tab', 'msno_source_system_tab_hist', ['msno_source_system_tab_count'])
            self.weighted_sum(features, 'source_type', 'msno_source_type_hist', ['msno_source_type_count'])
            members_concat_feats = [
                self.city, self.gender, self.registered_via, self.registration_init_time, self.expiration_date,
                self.msno_age_catg, self.msno_age_num, self.msno_tenure,
                self.msno_pos_query_hist_count,
                self.msno_neg_query_hist_count,
                self.msno_artist_name_hist_count, self.msno_artist_name_hist_mean,
                self.msno_composer_hist_count, self.msno_composer_hist_mean,
                self.msno_genre_ids_hist_count, self.msno_genre_ids_hist_mean,
                self.msno_language_hist_count, self.msno_language_hist_mean,
                self.msno_lyricist_hist_count, self.msno_lyricist_hist_mean,
                self.msno_source_screen_name_hist_count,
                self.msno_source_system_tab_hist_count,
                self.msno_source_type_hist_count
            ]
            self.members_feature = tf.concat(members_concat_feats, 1, name='members_feature')
            self.logger.info('self.members_feature: {}'.format(self.members_feature))

        with tf.variable_scope("songs") as scope:
            self.song_id = tf.nn.embedding_lookup(self.emb['song_id'], features['song_id'])
            self.language = tf.nn.embedding_lookup(self.emb['language'], features['language'])
            self.song_cc = tf.nn.embedding_lookup(self.emb['song_cc'], features['song_cc'])
            self.song_xxx = tf.nn.embedding_lookup(self.emb['song_xxx'], features['song_xxx'])
            self.song_yy = features['song_yy'][:, tf.newaxis]
            self.song_length = features['song_length'][:, tf.newaxis]
            self.song_pplrty = features['song_pplrty'][:, tf.newaxis]
            self.song_clicks = features['song_clicks'][:, tf.newaxis]

            self.song_weighted_sum(features, 'genre_ids', 'genre_ids', ['song_genre_ids_len'], is_seq=True)
            self.song_weighted_sum(features, 'artist_name', 'artist_name', ['song_artist_name_len'], is_seq=True)
            self.song_weighted_sum(features, 'composer', 'composer', ['song_composer_len'], is_seq=True)
            self.song_weighted_sum(features, 'lyricist', 'lyricist', ['song_lyricist_len'], is_seq=True)
            self.song_weighted_sum(features, 'city', 'song_city_hist', ['song_city_count', 'song_city_mean'])
            self.song_weighted_sum(features, 'gender', 'song_gender_hist', ['song_gender_count', 'song_gender_mean'])
            self.song_weighted_sum(features, 'msno_age_catg', 'song_msno_age_catg_hist',
                              ['song_msno_age_catg_count', 'song_msno_age_catg_mean'])
            self.song_weighted_sum(features, 'registered_via', 'song_registered_via_hist',
                              ['song_registered_via_count'])
            self.song_weighted_sum(features, 'source_screen_name', 'song_source_screen_name_hist',
                              ['song_source_screen_name_count'])
            self.song_weighted_sum(features, 'source_system_tab', 'song_source_system_tab_hist',
                              ['song_source_system_tab_count'])
            self.song_weighted_sum(features, 'source_type', 'song_source_type_hist',
                              ['song_source_type_count'])

            songs_concat_feats = [
                self.song_id, self.language, self.song_cc, self.song_xxx, self.song_yy,
                self.song_length, self.song_pplrty, self.song_clicks,
                self.genre_ids, self.artist_name,self.composer, self.lyricist,
                self.song_city_hist_count, self.song_city_hist_mean,
                self.song_gender_hist_count, self.song_gender_hist_mean,
                self.song_msno_age_catg_hist_count, self.song_msno_age_catg_hist_mean,
                self.song_registered_via_hist_count,
                self.song_source_screen_name_hist_count,
                self.song_source_system_tab_hist_count,
                self.song_source_type_hist_count
            ]
            self.songs_feature = tf.concat(songs_concat_feats, 1, name='songs_feature')
            self.logger.info('self.songs_feature: {}'.format(self.songs_feature))

        """'source_system_tab', 'source_screen_name', 'source_type'"""
        with tf.variable_scope("context"):
            self.source_system_tab = tf.nn.embedding_lookup(self.emb['source_system_tab'], features['source_system_tab'])
            self.source_screen_name = tf.nn.embedding_lookup(self.emb['source_screen_name'], features['source_screen_name'])
            self.source_type = tf.nn.embedding_lookup(self.emb['source_type'], features['source_type'])
            self.context_features = tf.concat(
                [self.source_system_tab, self.source_screen_name, self.source_type], 1, name='context_features')
            self.logger.info('self.context_features: {}'.format(self.context_features))

    def factor_encode(self, p, uniform_init_fn, has_context=True, mode=None, name='factor_mlp'):
        """Encode the concatenated features of members, songs, context, with fully connected layer

        :param uniform_init_fn: Initialization function
        :param has_context: Whether to handle context feature
        :param mode: The execution mode, as defined in tf.estimator.ModeKeys.
        :param name: Tensor name
        :return: Tensor list
        """
        is_train = mode == tf.estimator.ModeKeys.TRAIN
        ret = []
        with tf.variable_scope(name):
            factor_names = ('members_feature', 'songs_feature', 'context_features') if has_context else \
                      ('members_feature', 'songs_feature')
            factors = (self.members_feature, self.songs_feature, self.context_features) if has_context else \
                      (self.members_feature, self.songs_feature)
            for fc_name, factor in zip(factor_names, factors):
                hidden_units = self.construct_hidden_units(p, 'factor')
                self.logger.info('{} layers: {}'.format(fc_name, hidden_units))
                for layer in hidden_units:
                    factor = tf.layers.dense(factor, layer, kernel_initializer=uniform_init_fn,
                                    kernel_constraint=tf.keras.constraints.max_norm(),
                                    # kernel_regularizer=tf.contrib.layers.l2_regularizer(p.reg_scale),
                                    activation=tf.nn.selu)
                    # factor = tf.nn.relu(tf.layers.batch_normalization(factor))
                    if is_train and p.drop_rate > 0:
                        factor = alpha_dropout(factor, keep_prob=1 - p.drop_rate)

                ret.append(factor)
        return ret

    def model_fn(self, features, labels, mode, params):
        """Implement model graph

        :param features: A dictionary of tensors keyed by the feature name.
        :param labels: A tensor representing the labels
        :param mode: The execution mode, as defined in tf.estimator.ModeKeys.
        :return: `tf.estimator.EstimatorSpec`
        """
        p = pd.Series(params)
        is_train = mode == tf.estimator.ModeKeys.TRAIN
        self.logger.info('mode: {}, is_train: {}, '
                         'use dropout: {}'.format(mode, is_train, is_train and p.drop_rate > 0))

        self.base_features(p, features, labels, mode)

        uniform_init_fn = tf.glorot_uniform_initializer()
        self.members_feature, self.songs_feature, self.context_features = (
            self.factor_encode(p, uniform_init_fn, has_context=True, mode=mode, name='factor_mlp'))

        with tf.variable_scope("dnn", reuse=tf.AUTO_REUSE):
            net = tf.concat([self.members_feature, self.songs_feature, self.context_features], 1)
            self.logger.info('net: {}'.format(net))

            hidden_units = self.construct_hidden_units(p, 'dnn')
            self.logger.info('DNN layers: {}'.format(hidden_units))
            for layer in hidden_units:
                net = tf.layers.dense(net, layer, kernel_initializer=uniform_init_fn,
                                      kernel_constraint=tf.keras.constraints.max_norm(),
                                      # kernel_regularizer=tf.contrib.layers.l2_regularizer(p.reg_scale),
                                      activation=tf.nn.selu)
                # net = tf.nn.relu(tf.layers.batch_normalization(net))
                if is_train and p.drop_rate > 0:
                    net = alpha_dropout(net, 1 - p.drop_rate)
                    # net = tf.layers.dropout(net, rate=p.drop_rate, training=is_train)

            self.logits = tf.layers.dense(net, 1, kernel_initializer=uniform_init_fn, activation=None)
            self.pred = tf.nn.sigmoid(self.logits, name='pred')

        # Provide an estimator spec for `ModeKeys.PREDICT`
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                'outputs': tf.estimator.export.PredictOutput({
                    'predictions': self.pred
                })
            }
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=self.pred,
                                              export_outputs=export_outputs)

        with tf.variable_scope("loss"):
            self.labels = tf.to_float(labels[:, tf.newaxis])
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            for reg_term in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
                self.loss += reg_term
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope("metrics") as scope:
            self.auc = tf.metrics.auc(tf.cast(self.labels, tf.bool), self.pred)
            tf.summary.scalar('auc', self.auc[0])

        self.train_op = None
        self.global_step = tf.train.get_or_create_global_step()
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope("train"):
                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(update_ops):

                learning_rate = tf.train.cosine_decay(p.learning_rate,
                                                      self.global_step,
                                                      p.cos_decay_steps,
                                                      alpha=0.1)
                tf.summary.scalar("learning_rate", learning_rate)

                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)
                # self.train_op = tf.train.MomentumOptimizer(learning_rate, p.momentum)\
                #                         .minimize(self.loss, self.global_step)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops={'auc': self.auc},
            evaluation_hooks=[])

    def weighted_sum(self, features, dict_key, base_key, weighted_key: list):
        """Do weighted sum to multivalent column, just like tf.nn.embedding_lookup_sparse
          with combiner='sqrtn'.

        :param features: Data tensor from tf.data.Dataset api
        :param dict_key: Embedding key in model.emb
        :param base_key: Target variable to do weighted sum
        :param weighted_key: Weight variable key list, at least on must be provided
        :return: None
        """
        setattr(self, base_key, tf.nn.embedding_lookup(self.emb[dict_key], features[base_key]))

        for w_key in weighted_key:
            setattr(self, w_key, tf.nn.l2_normalize(features[w_key], 1)[:, :, tf.newaxis])

            tail = w_key.replace(base_key.replace('_hist', ''), '')
            name = '{}{}'.format(base_key, tail)
            val = tf.reduce_sum(getattr(self, base_key) * getattr(self, w_key), 1, name=name)
            setattr(self, name, val)

    def song_weighted_sum(self, features, dict_key, base_key, weighted_key: list, is_seq=False):
        """Do weighted sum to multivalent column of songs table, just like tf.nn.embedding_lookup_sparse
          with combiner='sqrtn'

        :param features: Data tensor from tf.data.Dataset api
        :param dict_key: Embedding key in model.emb
        :param base_key: Target variable to do weighted sum
        :param weighted_key: Weight variable key list, at least on must be provided
        :param is_seq: Is weights an array length description, if so,
        :return: None
        """
        if not is_seq:
            self.weighted_sum(features, dict_key, base_key, weighted_key)
        else:
            setattr(self, base_key, tf.nn.embedding_lookup(self.emb[dict_key], features[base_key]))

            w_key = weighted_key[0]
            setattr(self, w_key,
                    tf.nn.l2_normalize(tf.sequence_mask(features[w_key], dtype=tf.float32), 1)[:, :, tf.newaxis])

            name = base_key
            val = tf.reduce_sum(getattr(self, base_key) * getattr(self, w_key), 1, name=name)
            setattr(self, name, val)

    def get_embedding_var(self, shape, init_fn, name, zero_first=True):
        """Initialize the embedding variable, `zero_first` indicate whether to reserve a zero vector
          for not trainable parameters, otherwise treat outlier a special category.

        :param shape: Shape of [unique value of categorical feature, embedding dimension]
        :param init_fn: Initialization funciton
        :param name: Tensor name
        :param zero_first: Indicate to prepend zero vector to embedding variables(not trainable)
        :return: Initialized embedding variable
        """
        if zero_first:
            return tf.concat([
                tf.Variable(tf.zeros([1, shape[1]]), trainable=False),
                tf.Variable(init_fn(shape=shape), name=name)
            ], 0)
        else:
            return tf.Variable(init_fn(shape=[shape[0] + 1, shape[1]]), name=name)


    def get_estimator(self, p, config:tf.estimator.RunConfig):
        """Customized tf.estimator.Estimator object

        ;param p: Received parameter, type pd.Series
        :param config: see `tf.estimator.RunConfig`
        :return: `tf.estimator.Estimator`
        """
        self.logger.info('creating a custom Estimator')
        est = tf.estimator.Estimator(model_fn=self.model_fn, model_dir=self.model_dir,
                                     config=config, params=p.to_dict())

        # Create directory for export, it will raise error if in GCS environment
        try:
            tf.gfile.MakeDirs('{}/export/{}'.format(p.model_dir, p.export_name), exist_ok=True)
        except Exception as e:
            self.logger.warn(e)
        return est

    def construct_hidden_units(self, p, part='dnn'):
        """Create the number of hidden units in each layer

        it will use a "decay" mechanism to define the number of units in each layer.

        :param p: Received parameters
        :param part: 'dnn' or 'factor', 'dnn' for mlp (DNN) part, 'factor' for matrix factorization part
        :return:
        """

        first_layer_size = p.first_mlp_layer_size if part == 'dnn' else p.first_factor_layer_size
        hidden_units = [
            max(2, int(first_layer_size * p.scale_factor ** i))
            for i in range(p.num_layers)
        ]

        # self.logger.info("Hidden units structure: {}".format(hidden_units))
        return hidden_units

class NeuMFModel(Model):
    """Inherit Model(DNN), implement this paper [NCF](https://arxiv.org/pdf/1708.05031.pdf)"""
    def __init__(self, *args, **kwargs):
        super(NeuMFModel, self).__init__(*args, **kwargs)

    def model_fn(self, features, labels, mode, params):
        """Implement model graph

        :param features: A dictionary of tensors keyed by the feature name.
        :param labels: A tensor representing the labels
        :param mode: The execution mode, as defined in tf.estimator.ModeKeys.
        :return: `tf.estimator.EstimatorSpec`
        """
        p = pd.Series(params)
        is_train = mode == tf.estimator.ModeKeys.TRAIN
        self.logger.info('mode: {}, is_train: {}, '
                         'use dropout: {}'.format(mode, is_train, is_train and p.drop_rate > 0))

        self.base_features(p, features, labels, mode)

        uniform_init_fn = tf.glorot_uniform_initializer()
        mlp_members, mlp_songs, mlp_conext = (
            self.factor_encode(p, uniform_init_fn, has_context=True, mode=mode, name='factor_mlp'))

        with tf.variable_scope("mlp", reuse=tf.AUTO_REUSE):
            self.mlp_vector = tf.concat([mlp_members, mlp_songs, mlp_conext], 1)
            hidden_units = self.construct_hidden_units(p, 'dnn')
            self.logger.info('MLP layers: {}'.format(hidden_units))
            for layer in hidden_units:
                self.mlp_vector = tf.layers.dense(self.mlp_vector, layer,
                                      kernel_initializer=uniform_init_fn,
                                      kernel_constraint=tf.keras.constraints.max_norm(),
                                      # kernel_regularizer=tf.contrib.layers.l2_regularizer(p.reg_scale),
                                      activation=tf.nn.selu)
                # self.mlp_vector = tf.nn.relu(tf.layers.batch_normalization(self.mlp_vector))
                if is_train and p.drop_rate > 0:
                    # self.mlp_vector = tf.layers.dropout(self.mlp_vector, p.drop_rate, training=is_train)
                    self.mlp_vector = alpha_dropout(self.mlp_vector, keep_prob=1 - p.drop_rate)

        mf_members, mf_songs = self.factor_encode(p, uniform_init_fn, has_context=False, mode=mode, name='factor_mf')
        with tf.variable_scope("mf", reuse=tf.AUTO_REUSE):
            self.mf_vector = tf.multiply(mf_members, mf_songs)
            # self.mf_vector = tf.nn.relu(tf.layers.batch_normalization(self.mf_vector))
            if is_train and p.drop_rate > 0:
                self.mf_vector = tf.layers.dropout(self.mf_vector, rate=p.drop_rate, training=is_train)

        with tf.variable_scope("concatenate", reuse=tf.AUTO_REUSE):
            self.net = tf.concat([self.mf_vector, self.mlp_vector], 1)
            self.logits = tf.layers.dense(self.net, 1, kernel_initializer=uniform_init_fn,
                            kernel_constraint=tf.keras.constraints.max_norm(),
                            # kernel_regularizer=tf.contrib.layers.l2_regularizer(p.reg_scale),
                            activation=None)
            self.pred = tf.nn.sigmoid(self.logits)

        # Provide an estimator spec for `ModeKeys.PREDICT`
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {
                'outputs': tf.estimator.export.PredictOutput({
                    'predictions': self.pred
                })
            }
            return tf.estimator.EstimatorSpec(mode=mode,
                                              predictions=self.pred,
                                              export_outputs=export_outputs)

        with tf.variable_scope("loss"):
            self.labels = tf.to_float(labels[:, tf.newaxis])
            self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.logits))
            for reg_term in tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES):
                self.logger.info('reg_term: {}'.format(reg_term.name))
                self.loss += reg_term
            tf.summary.scalar('loss', self.loss)

        with tf.variable_scope("metrics") as scope:
            self.auc = tf.metrics.auc(tf.cast(self.labels, tf.bool), self.pred)
            tf.summary.scalar('auc', self.auc[0])

        self.train_op = None
        self.global_step = tf.train.get_or_create_global_step()
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.variable_scope("train"):
                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                # with tf.control_dependencies(update_ops):

                learning_rate = tf.train.cosine_decay(p.learning_rate,
                                                      self.global_step,
                                                      p.cos_decay_steps,
                                                      alpha=0.1)
                tf.summary.scalar("learning_rate", learning_rate)
                self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, global_step=self.global_step)
                # self.train_op = tf.train.GradientDescentOptimizer(learning_rate)\
                #                         .minimize(self.loss, global_step=self.global_step)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=self.loss,
            train_op=self.train_op,
            eval_metric_ops={'auc': self.auc},
            evaluation_hooks=[])


class BestScoreExporter(tf.estimator.Exporter):
    """Only export model has the best score, like lowest loss or highest accuracy."""
    logger = utils.logger('BestScoreExporter')

    def __init__(self,
                 name,
                 serving_input_receiver_fn,
                 assets_extra=None,
                 as_text=False,
                 model_dir=None):
        self._name = name
        self.serving_input_receiver_fn = serving_input_receiver_fn
        self.assets_extra = assets_extra
        self.as_text = as_text
        self.model_dir = model_dir
        self.best = self.get_last_eval()
        self._exports_to_keep = 1
        self.export_result = None
        self.logger.info('BestScoreExporter init, last best eval is {}'.format(self.best))

    @property
    def name(self):
        return self._name

    def get_last_eval(self):
        """Get the latest best score."""
        path = '{}/best.eval'.format(self.model_dir)
        if tf.gfile.Exists(path):
            return utils.read_pickle(path)
        else:
            return None

    def save_last_eval(self, best:float):
        """Save the latest best score."""
        self.logger.info('Persistent best eval: {}'.format(best))
        path = '{}/best.eval'.format(self.model_dir)
        utils.write_pickle(path, best)

    def export(self, estimator, export_path, checkpoint_path, eval_result,
             is_the_final_export):
        """Call `tf.estimator.Estimator.export_savedmodel` to export model to protocol buffer.

        :param estimator: `tf.estimator.Estimator` object
        :param export_path: A string containing a directory where to write the export.
        :param checkpoint_path: The checkpoint path to export.
        :param eval_result: The output of `Estimator.evaluate` on this checkpoint.
        :param is_the_final_export: This boolean is True when this is an export in the
            end of training.  It is False for the intermediate exports during
            the training.
        :return:
        """
        self.logger.info('eval_result: {}'.format(eval_result))

        curloss = eval_result['loss']
        if self.best is None or self.best >= curloss:
            # Clean first, only keep the best weights
            export_path = export_path.replace('\\', '/')
            self.logger.info('clean export_path: {}'.format(export_path))
            try:
                tf.gfile.DeleteRecursively(export_path)
            except Exception as e:
                self.logger.warn(e)

            tf.gfile.MakeDirs(export_path)

            self.best = curloss
            self.save_last_eval(self.best)

            self.logger.info('nice eval loss: {}, export to pb'.format(curloss))
            self.export_result = estimator.export_savedmodel(
                export_path,
                self.serving_input_receiver_fn,
                assets_extra=self.assets_extra,
                as_text=self.as_text,
                checkpoint_path=checkpoint_path)
        else:
            self.logger.info('bad eval loss: {}'.format(curloss))

        return self.export_result
