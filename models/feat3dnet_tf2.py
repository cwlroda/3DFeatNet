import logging
import tensorflow as tf
from tensorflow._api.v2 import train
from tensorflow.python.ops.gen_state_ops import assign_add_eager_fallback
from tensorflow.python.ops.numpy_ops.np_math_ops import positive

from models.pointnet_common import sample_points, sample_and_group, sample_and_group_all, query_and_group_points
from models.layers_tf2 import pairwise_dist, MaxPoolAxis, MaxPoolConcat

class PointnetSaModule(tf.Module):
    """ PointNet Set Abstraction (SA) Module. Modified to remove unneeded components (e.g. pooling),
        normalize points based on radius, and for a third layer of MLP
    """
    
    def __init__(self, mlp, mlp2, mlp3, name, bn=True, final_relu=True):
        '''
        Args:
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP after max pooling concat
            mlp3: list of int32 -- output size for MLP after second max pooling
            is_training (tf.placeholder): Indicate training/validation
            name (str): name scope
            bn (bool): Whether to perform batch normalizaton
            bn_decay: Decay schedule for batch normalization
            final_relu: Whether to use relu as the final activation function
        '''
        super(PointnetSaModule, self).__init__(name=name)
        # Initialise layers
        self.layers = []
        
        # Define mlp layers based on input dimension
        for i, num_out_channel in enumerate(mlp):
            self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                        padding="valid", name='conv_%d' %(i), activation='relu'
                                        )
                              )
            if bn:
                # TODO figure out the appropriate axis. Rest are set to default.
                self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn" ) )

        # Max pool, then concatenate
        self.layers.append( MaxPoolConcat() ) # Custom layer pooling only on one axis then tiling then concat

        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                        padding="valid", name='conv_mid_%d' %(i),
                                        activation=tf.nn.relu if (final_relu or i < len(mlp2) - 1) else None
                                        )
                                  )
                if bn:
                    # TODO figure out the appropriate axis. Rest are set to default.
                    self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn_mid" ) )
        
        # Max pool again
        self.layers.append( MaxPoolAxis() )

        if mlp3 is not None:
            for i, num_out_channel in enumerate(mlp3):
                self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                        padding="valid", name='conv_post_%d' %(i),
                                        activation=tf.nn.relu if (final_relu or i < len(mlp2) - 1) else None
                                        )
                                  )
                if bn:
                    # TODO figure out the appropriate axis. Rest are set to default.
                    self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn_post" ) )

    @tf.Module.with_name_scope
    def __call__(self, xyz: tf.Tensor, points: tf.Tensor, npoint: int, radius: float, nsample,
                    is_training, tnet_spec=None, knn=False, use_xyz=True, 
                    keypoints=None, orientations=None, normalize_radius=True):
        """
        Args:
            xyz (tf.Tensor): (batch_size, ndataset, 3) TF tensor
            points (tf.Tensor): (batch_size, ndataset, num_channel)
            npoint (int32): #points sampled in farthest point sampling
            radius (float): search radius in local region
            nsample (int): Maximum points in each local region
            is_training (bool): calling train or inference
            tnet_spec: Unused in Feat3D-Net. Set to None
            knn: Unused in Feat3D-Net. Set to False
            use_xyz: Unused in Feat3D-Net. Set to True
            keypoints: If provided, cluster centers will be fixed to these points (npoint will be ignored)
            orientations (tf.Tensor): Containing orientations from the detector
            normalize_radius (bool): Whether to normalize coordinates [True] based on cluster radius.

        Returns:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
        """

        if npoint is None:
            self.nsample = xyz.get_shape()[1]   # Number of samples
            self.new_xyz, self.new_points, self.idx, self.grouped_xyz = \
                sample_and_group_all(xyz, points, use_xyz)
        else:
            self.new_xyz, self.new_points, self.idx, self.grouped_xyz, self.end_points = \
                sample_and_group(npoint, radius, nsample, xyz, points, tnet_spec,
                            knn, use_xyz, keypoints=keypoints,
                            orientations=orientations,
                            normalize_radius=normalize_radius)

        for layer in self.layers:
            self.new_points = layer(self.new_points, training=is_training)

        self.new_points = tf.squeeze(self.new_points, [2])

        return self.new_xyz, self.new_points, self.idx, self.end_points

# TODO: Check if defining the gradients within this module (translate from tf1 to tf2...) is reqd
class FeatureDetectionModule(tf.Module):
    """Detect features in point cloud.
    
    `compute_det_gradients` is experimental?
    """

    def __init__(self, mlp, mlp2, name, bn=True):
        """
        Args:
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP on each region. Set to None or [] to ignore
        name: label
        bn: bool -- Whether to perform batch normalization

        """
        super(FeatureDetectionModule, self).__init__(name=name)

        self.end_points = {}
        self.end_points['gradients'] = {}
        self.end_points['gradients']['det'] = {}

        self.layers = []
        
        # Define mlp layers based on input dimension
        for i, num_out_channel in enumerate(mlp):
            self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                        padding="valid", name='conv_%d' %(i), activation='relu' )
                              )
            if bn:
                # TODO figure out the appropriate axis. Rest are set to default.
                self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn" ) )

        # Max pool, then concatenate
        self.layers.append( MaxPoolAxis() ) # Custom layer pooling only on one axis then tiling then concat

        if mlp2 is not None:
            for i, num_out_channel in enumerate(mlp2):
                self.layers.append( tf.keras.layers.Conv2D( num_out_channel, kernel_size=[1,1], strides=[1,1],
                                        padding="valid", name='conv_mid_%d' %(i)
                                    )
                                  )
                if bn:
                    # TODO figure out the appropriate axis. Rest are set to default.
                    self.layers.append( tf.keras.layers.BatchNormalization( axis=-1, name="bn_mid" ) )

        # Two "endpoints" of the calculation
        self.attention = tf.keras.layers.Conv2D(1, kernel_size=[1,1], strides=[1,1], padding='valid',
                                                activation='softplus', name="attention"
                                               )

        self.orientation = tf.keras.layers.Conv2D(2, kernel_size=[1,1], strides=[1,1], padding='valid',
                                                    name='orientation' )
    
    @tf.Module.with_name_scope
    def __call__(self, xyz, points, num_clusters, radius, is_training, num_samples=64, compute_det_gradients=True):
        """
        Args:
        xyz (tf.Tensor): Input point cloud of size (batch_size, ndataset, 3)
        points (tf.Tensor): Point features. Unused in 3DFeat-Net
        num_clusters (int): Number of clusters to extract. Set to -1 to use all points
        radius (float): Radius to consider for feature detection
        num_samples: Maximum number of points to consider per cluster
        is_training (bool): train or inference
        compute_det_gradients (bool): if gradients are meant to be calculated

        Returns:
        new_xyz: Cluster centers
        idx: Indices of points sampled for the clusters
        attention: Output attention weights
        orientation: Output orientation (radians)
        """

        new_xyz = sample_points(xyz, num_clusters)  # Sample point centers
        new_points, idx = query_and_group_points(xyz, points, new_xyz, num_samples, radius, knn=False, 
                            use_xyz=True, normalize_radius=True, orientations=None)  # Extract clusters

        last_layer = 0

        for layer in self.layers:
            new_points = layer(new_points, training=is_training)
            
            if layer.name[:4] == "conv" and compute_det_gradients:
                # TODO This might not work out of the box
                self.end_points['gradients']['det']['mlp_{}'.format(last_layer)] = \
                    tf.gradients(new_points, xyz, new_points)

                last_layer += 1
        
        # Attention and orientation regression
        attention_out = self.attention(new_points)
        attention = tf.squeeze(attention_out, axis=[2, 3])

        orientation_xy = self.orientation(new_points)
        orientation_xy = tf.squeeze(orientation_xy, axis=2)
        orientation_xy = tf.nn.l2_normalize(orientation_xy, axis=2, epsilon=1e-8)
        orientation = tf.atan2(orientation_xy[:, :, 1], orientation_xy[:, :, 0])

        return new_xyz, idx, attention, orientation, self.end_points


class FeatureExtractionModule(PointnetSaModule):
    """ Extract feature descriptors """

    def __init__(self, mlp, mlp2, mlp3, name="layer1", bn=True):
        '''
        Args:
        is_training (tf.placeholder): Set to True if training, False during evaluation
        mlp: list of int32 -- output size for MLP on each point
        mlp2: list of int32 -- output size for MLP after max pooling concat
        mlp3: list of int32 -- output size for MLP after second max pooling
        '''
        super(FeatureExtractionModule, self).__init__(
            mlp, mlp2, mlp3, name, bn, final_relu=False
        )
    
    @tf.Module.with_name_scope
    def __call__(self, l0_xyz: tf.Tensor, l0_points: tf.Tensor, 
                radius, num_samples, keypoints, orientations, is_training):
        '''
        Args:
        l0_xyz (tf.Tensor): Input point cloud of size (batch_size, ndataset, 3)
        l0_points (tf.Tensor): Point features. Unused in 3DFeat-Net
        radius: Radius to consider for feature detection
        num_samples: Maximum points in each local region
        keypoints: Keypoints to compute features for
        orientations: Orientation (from detector) to pre-rotate clusters before compute descriptors
        is_training (bool): train or infer

        Returns:
        xyz, features, end_points
        '''
                
        l1_xyz, l1_points, l1_idx, end_points = \
        super(FeatureExtractionModule, self).__call__(
            l0_xyz, l0_points, 512, radius, num_samples, is_training, keypoints=keypoints,
            orientations=orientations, normalize_radius=True
        )

        features = tf.nn.l2_normalize(l1_points, axis=2, epsilon=1e-8)
        
        return l1_xyz, features, end_points


class Feat3dNetInference(tf.Module):
    '''
    Contains operations specific to the Inference model of 3DFeatNet.
    Refactored from `Feat3dNet.get_inference_model()`
    '''
    def __init__(self, det_mlps: 'dict', ext_mlps: 'dict', params: 'dict', name, bn=True):
        """ Constructs the core 3DFeat-Net model.

        Args:
            det_mlps(dict< str, list<int> >): key: 'mlp*' corresponding to the list reqd for the respective mlp
            ext_mlps(dict< str, list<int> >): same as det_mlps
            params (dict): Passes in parameters from outside class for Detection and Extraction
            name (str)
            bn (bool): Whether to perform batch normalization

        Returns:
            xyz, features, attention, end_points

        """
        super(Feat3dNetInference, self).__init__(name=name)

        self.Detection = FeatureDetectionModule(det_mlps['mlp'], det_mlps['mlp2'], 
                                "Feature_Det_Module", bn)
        self.Extraction = FeatureExtractionModule(ext_mlps['mlp'], ext_mlps['mlp2'], 
                                ext_mlps['mlp3'], "Feature_Ext_Module", bn)

        self._num_clusters = params['num_clusters']
        self._radius = params['BaseScale']
        self._num_samples = params['num_samples']
        self._NoRegress = params['NoRegress']
        self._Attention = params['Attention']
    
    @tf.Module.with_name_scope
    def __call__(self, point_cloud, is_training, compute_det_gradients=True):
        '''
        Args:
            point_cloud (tf.Tensor): Input point clouds of size (batch_size, ndataset, 3).
            is_training (bool): Training or Inference
            compute_det_gradients (bool): Whether to compute and output gradients for the the Detection stage.

        Returns:
            xyz, features, attention, end_points
        '''

        l0_xyz = point_cloud[:, :, :3]
        l0_points = None    # Normal information not used in 3DFeat-Net
        end_points = {}

        keypoints, idx, attention, orientation, end_points_temp = \
            self.Detection(l0_xyz, l0_points, self._num_clusters, self._radius, 
                            is_training, self._num_samples, compute_det_gradients)

        end_points.update(end_points_temp)
        end_points['keypoints'] = keypoints
        end_points['attention'] = attention
        end_points['orientation'] = orientation

        keypoint_orientation = orientation

        if self._NoRegress:
            keypoint_orientation = None
        if not self._Attention:
            attention = None

        xyz, features, endpoints_temp = \
            self.Extraction(l0_xyz, l0_points, keypoints=keypoints, orientations=keypoint_orientation,
                                    radius=self.param['BaseScale'], is_training=is_training, 
                                    num_samples=self.param['num_samples'])

        end_points.update(endpoints_temp)

        return xyz, features, attention, end_points
    

class Feat3dNetTrain(Feat3dNetInference):
    '''
    Contains operations specific to the Inference model of 3DFeatNet.
    Refactored from `Feat3dNet.get_train_model()`, and thus inherits from `Feat3dNetInference`.
    '''
    def __init__(self, det_mlps: 'dict', ext_mlps: 'dict', params, name, bn=True):
        '''
        Constructs the training model. Essentially calls `get_inference_model`, but
        also handles the training triplets.

        Args:
            det_mlps(dict< str, list<int> >): key: 'mlp*' corresponding to the list reqd for the respective mlp
            ext_mlps(dict< str, list<int> >): same as det_mlps
            params (dict): Passes in parameters from outside class for Detection and Extraction
            name (str)
            bn (bool): Whether to perform batch normalization

        '''
        super(Feat3dNetTrain, self).__init__(det_mlps, ext_mlps, params, name, bn)

        self._params = params
    
    @tf.Module.with_name_scope
    def __call__(self, anchors, positives, negatives, is_training, compute_det_gradients=True):
        '''
        Args:
            anchors (tf.Tensor): Anchor point clouds of size (batch_size, ndataset, 3).
            positives (tf.Tensor): Positive point clouds, same size as anchors
            negatives (tf.Tensor): Negative point clouds, same size as anchors
            is_training (bool): Train vs infer
            compute_det_gradients (bool): Whether to compute and output gradients for the the Detection stage.
        Returns:
            xyz, features, anchor_attention, end_points
        '''
        end_points = {}

        point_clouds = tf.concat([anchors, positives, negatives], axis=0)
        end_points['input_pointclouds'] = point_clouds

        xyz, features, attention, endpoints_temp = \
            super(Feat3dNetTrain, self).__call__(point_clouds, is_training, compute_det_gradients)

        end_points['output_xyz'] = xyz
        end_points['output_features'] = features
        end_points.update(endpoints_temp)

        xyz = tf.split(xyz, 3, axis=0, name="xyz")
        features = tf.split(features, 3, axis=0, name="features")
        anchor_attention = tf.split(attention, 3, axis=0, 
            name="anchor_attention")[0] if attention is not None else None

        return xyz, features, anchor_attention, end_points


class Feat3dNet(tf.keras.Model):
    def __init__(self, train_or_infer, name="3DFeatNet", param=None):
        """ Constructor: Creates the 3dFeatNet model by calling its relevant sub-objects.

        Args:
            train_or_infer (bool): Whether to call `get_inference_model` (false) 
                                   or `get_train_model` (true)

            param:    Python dict containing the algorithm parameters. It should contain the
                        following fields (square brackets denote paper's parameters':
                        'NoRegress': Whether to skip regression of the keypoint orientation.
                                    [False] (i.e. regress)
                        'BaseScale': Cluster radius. [2.0] (as in the paper)
                        'Attention': Whether to predict the attention. [True]
                        'num_clusters': Number of clusters [512]
                        'num_samples': Maximum number of points per cluster [64]
                        'margin': Triplet loss margin [0.2]
        """
        super(Feat3dNet, self).__init__(name=name)

        self.logger = logging.getLogger(self.__class__.__name__)
        self.param = {}
        self.param.update(param)
        self.logger.info('Model parameters: %s', self.param)

        det_mlps = {'mlp': [64, 128, 256], 'mlp2': [128, 64]}

        # Descriptor extraction: Extract descriptors for each cluster
        mlp = [32, 64]
        mlp2 = [128] if self.param['feature_dim'] <= 64 else [256]
        mlp3 = [self.param['feature_dim']]

        ext_mlps = {'mlp': mlp, 'mlp2': mlp2, 'mlp3': mlp3}

        self.logger.info('Descriptor MLP sizes: {} | {} | {}'.format(mlp, mlp2, mlp3))

        self.train_or_infer = train_or_infer
        if train_or_infer: 
            self.Network = Feat3dNetTrain(det_mlps, ext_mlps, self.param, "Feat3dNet")
        else: 
            self.Network = Feat3dNetInference(det_mlps, ext_mlps, self.param, "Feat3dNet")

    def call(self, inputs: 'list[tf.keras.Input]', training=False):
        if self.train_or_infer:
            anchors, positives, negatives = inputs
            # anchors = inputs['anchors']
            # positives = inputs['positives']
            # negatives = inputs['negatives']

            return self.Network(anchors, positives, negatives, training)
        else:
            point_cloud = inputs
            # point_cloud = inputs['point_cloud']

            return self.Network(point_cloud, training)


    # def compile(self, optimizer, loss, metrics, loss_weights, weighted_metrics, 
    #     run_eagerly, steps_per_execution, **kwargs):
    #     '''
    #     Configures the model for training.
    #     '''

    #     return super().compile(optimizer=optimizer, loss=loss, 
    #             metrics=metrics, loss_weights=loss_weights, 
    #             weighted_metrics=weighted_metrics, run_eagerly=run_eagerly,
    #             steps_per_execution=steps_per_execution, **kwargs)


    # def evaluate(self, x, y, batch_size, verbose, sample_weight, steps, 
    #             callbacks, max_queue_size, workers, use_multiprocessing, 
    #             return_dict, **kwargs):
    #     '''
    #     Returns the loss value & metrics values for the model in test mode.
    #     Computation is done in batches (see `batch_size`)
    #     '''
    #     return super().evaluate(x=x, y=y, batch_size=batch_size, verbose=verbose, 
    #             sample_weight=sample_weight, steps=steps, callbacks=callbacks,
    #             max_queue_size=max_queue_size, workers=workers, use_multiprocessing=use_multiprocessing, 
    #             return_dict=return_dict, **kwargs)

    @tf.function
    def feat_3d_net_loss( self, y_true, y_pred ):
        """ 
        Computes the attention weighted alignment loss as described in our paper.

        Args:
            y_true: Attention from anchor point clouds
            y_pred: List of [anchor_features, positive_features, negative_features]

        Returns:
            loss (scalar)
        """
        anchors, positives, negatives = y_pred

        # Computes for each feature of the anchor, the distance to the nearest feature in the positive and negative
        positive_dist = pairwise_dist(anchors, positives)
        negative_dist = pairwise_dist(anchors, negatives)
        best_positive = tf.reduce_min(positive_dist, axis=2)
        best_negative = tf.reduce_min(negative_dist, axis=2)
 
        if not self.param['Attention']:
            sum_positive = tf.reduce_mean(best_positive, 1)
            sum_negative = tf.reduce_mean(best_negative, 1)
        else:
            attention_sm = y_true / tf.reduce_sum(y_true, axis=1)[:, None]
            sum_positive = tf.reduce_sum(attention_sm * best_positive, 1)
            sum_negative = tf.reduce_sum(attention_sm * best_negative, 1)

            # tf.compat.v1.summary.histogram('normalized_attention', attention_sm)
            tf.summary.histogram("normalized_attention", attention_sm)
            triplet_cost = tf.maximum(0., sum_positive - sum_negative + self.param['margin'])

            loss = tf.reduce_mean(triplet_cost)

        # tf.compat.v1.summary.scalar('loss', loss)    # Force using the TF1 version of scalar summary
        tf.compat.v1.summary.scalar('loss', loss)

        return loss

    # @tf.function
    # def get_train_op(self, loss_op, lr=1e-5, global_step=None):
    #     """ 
    #     Gets training op
    #     """
    #     optimizer = tf.keras.optimizers.Adam(lr)


    #     var_list = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)

    #     to_exclude = []
    #     if self.param['freeze_scopes'] is not None:
    #         for s in self.param['freeze_scopes']:
    #             to_exclude += tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=s)
    #     var_list = [v for v in var_list if v not in to_exclude]

    #     train_op = optimizer.minimize(loss_op, global_step=global_step,
    #                                   var_list=var_list)
    #     return train_op
