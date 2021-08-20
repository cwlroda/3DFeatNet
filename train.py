import argparse
import coloredlogs, logging
import logging.config
import numpy as np
import os
import sys
import tensorflow as tf

from models.feat3dnet import Feat3dNet, AttentionWeightedAlignmentLoss

from config import *
from data.datagenerator import DataGenerator
from data.augment import get_augmentations_from_list
from utils import get_tensors_in_checkpoint_file

NUM_CLUSTERS = 512
UPRIGHT_AXIS = 2  # Will learn invariance along this axis
VAL_PROPORTION = 1.0

# Arguments
parser = argparse.ArgumentParser(description='Trains 3dFeatNet_tf2')
parser.add_argument('--gpu', type=str, default=0,
                    help='GPU to use (default: None (Distributed Training strategy))')
# data
parser.add_argument('--data_dim', type=int, default=6,
                    help='Input dimension for data. Note: Feat3D-Net will only use the first 3 \
                    dimensions (default: 6)')
parser.add_argument('--data_dir', type=str, default='data/oxford',
                    help='Path to dataset. Should contain "train" and "clusters" folders')
# Model
parser.add_argument('--model', type=str, default='3DFeatNet', help='Model to load')
parser.add_argument('--noregress', action='store_true',
                    help='If set, regression of feature orientation will not be performed')
parser.add_argument('--noattention', action='store_true',
                    help='If set, model will not learn to predict attention')
parser.add_argument('--margin', type=float, default=0.2,
                    help='Margin for triplet loss. Default=0.2')
parser.add_argument('--feature_dim', type=int, default=32, choices=[16, 32, 64, 128],
                    help='Feature dimension size')
# Data
parser.add_argument('--num_points', type=int, default=4096,
                    help='Number of points to downsample model to')
parser.add_argument('--base_scale', type=float, default=2.0,
                    help='Base scale (radius) for sampling clusters. Set to around 2.0 for oxford dataset')
parser.add_argument('--num_samples', type=int, default=64,
                    help='Maximum number of points to consider per cluster (default: 64)')
parser.add_argument('--augmentation', type=str, nargs='+', default=['Jitter', 'RotateSmall', 'Shift', 'Rotate1D'],
                    choices=['Jitter', 'RotateSmall', 'Rotate1D', 'Rotate3D', 'Scale', 'Shift'],
                    help='Data augmentation settings to use during training')
# Logging
parser.add_argument('--log_dir', type=str, default='./ckpt',
                    help='Directory to save tf summaries, checkpoints, and log. Default=./ckpt')
parser.add_argument('--ignore_missing_vars', action='store_true',
                    help='Whether to crash if variables are missing')
parser.add_argument('--summary_every_n_steps', type=int, default=20,
                    help='Save to tf summary every N steps. Default=20')
parser.add_argument('--validate_every_n_steps', type=int, default=250,
                    help='Run over validation data every n steps. Default=250')
# Saver
parser.add_argument('--checkpoint', type=str,
                    help='Checkpoint to restore from (optional)')
parser.add_argument('--checkpoint_every_n_steps', type=int, default=500,
                    help='Save a checkpoint every n steps. Default=500')
parser.add_argument('--restore_exclude', type=str, nargs='+', default=None,
                    help='To ignore from checkpoint')
# Training
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='Number of epochs to train for. Default=1000')
args = parser.parse_args()

# Prepares the folder for saving checkpoints, summary, logs, SavedModels
log_dir = args.log_dir
checkpoint_dir = os.path.join(log_dir, 'ckpt')
model_savepath = os.path.join(log_dir, "SavedModel")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(model_savepath, exist_ok=True)

# Create Logging
logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

fileHandler = logging.FileHandler("{0}/log.txt".format(log_dir))
logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)


def log_arguments():
    logger.info('Command: %s', ' '.join(sys.argv))

    s = '\n'.join(['    {}: {}'.format(arg, getattr(args, arg)) for arg in vars(args)])
    s = 'Arguments:\n' + s
    logger.info(s)

def train(gpu_list):
    '''
    Overall wrapper function for training the 3DFeatNet model.
    If multi-GPU behaviour is desired, do not enable this flag.
    '''

    ### BEGIN INIT STUFF ###

    logger.info("Saving logs in {}".format(log_dir))
    logger.info("Saving checkpoints in {}".format(checkpoint_dir))
    logger.info("Saving SavedModels in {}".format(model_savepath))

    log_arguments()

    # init training data
    train_file = os.path.join(args.data_dir, 'train/train.txt')
    train_data = DataGenerator(train_file, num_cols=args.data_dim)

    logger.info('Loaded train data: %s (# instances: %i)',
                train_file, train_data.size)
    
    train_augmentations = get_augmentations_from_list(args.augmentation, upright_axis=UPRIGHT_AXIS)

    # init validation data
    val_folder = os.path.join(args.data_dir, 'clusters')
    val_groundtruths = load_validation_groundtruths(os.path.join(val_folder, 'filenames.txt'),
                                                    proportion=VAL_PROPORTION)

    # Model
    param = {'NoRegress': args.noregress, 'BaseScale': args.base_scale, 'Attention': not args.noattention,
             'margin': args.margin, 'num_clusters': NUM_CLUSTERS, 'num_samples': args.num_samples,
             'feature_dim': args.feature_dim, 'freeze_scopes': None,
             }

    # Distributed training strategy:
    mirrored_strat = tf.distribute.MirroredStrategy(devices=gpu_list)   # Make use of all GPUs on the device

    with mirrored_strat.scope():
        model = Feat3dNet(True, param=param)

        # Need to put in a dummy input to initialize the model.
        rand_input = tf.concat([tf.random.normal([BATCH_SIZE, args.num_points, args.data_dim])]*3, axis=0)
        model(rand_input, training=True)

        # Extract initialised weights out of the model, for restoration later.
        raw_weights = {}
        for layer in model.layers:
            raw_weights[layer.name] = layer.get_weights()

        # init model
        model_find = tf.train.latest_checkpoint(checkpoint_dir)
        if model_find is not None:
            model.load_weights(model_find)
            logger.info('Restored weights from {}.'.format(model_find))
        else:
            logger.info('Unable to find a latest checkpoint in {}.'.format(checkpoint_dir))
        loss_fn = AttentionWeightedAlignmentLoss(param['Attention'], param['margin'])
        optimizer = tf.keras.optimizers.Adam(1e-5)

    # Reset model weights if necessary. Does a simple nested search.
    if args.restore_exclude is not None:
        logger.info("Found layers to regenerate weights for in restore_exclude: {}".format(args.restore_exclude))
        for layer in model.layers:
            for x in args.restore_exclude:
                if x in layer.name:
                    logger.info("In layer: {}".format(layer.name))

                    print("Before resetting:")
                    for weight in layer.trainable_weights:
                        print("\t{}\t{}".format(weight.name, weight.numpy()[:5]))

                    layer.set_weights( raw_weights[layer.name] )

                    print("After resetting:")
                    for weight in layer.trainable_weights:
                        print("\t{}\t{}".format(weight.name, weight.numpy()[:5]))

    else:
        logger.info("No weights to regenerate in restore_exclude.")

    del raw_weights # Save memory

    # init summary writers
    logger.info('Summaries will be stored in: %s', args.log_dir)
    train_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, 'train'))
    test_writer = tf.summary.create_file_writer(os.path.join(args.log_dir, 'test'))
    train_writer.init()
    test_writer.init()

    logger.info('Training Batch size: %i, validation batch size: %i', BATCH_SIZE, VAL_BATCH_SIZE)
    logger.info("TF Executing Eagerly? {}".format(tf.executing_eagerly() ))  # Shld be true

    # debug here
    # print("Model has {} trainable weights".format(len(model.trainable_weights)))
    # for w in model.trainable_weights:
    #     print(w.name, w.shape, w.dtype)

    model.summary(print_fn=logger.info)
    # input(">>>Continue?<<<") # breakpoint
    ### END INIT STUFF ###

    step = 0
    for iEpoch in range(args.num_epochs):
        logger.info('Starting epoch %d.' %iEpoch)

        train_data.shuffle()
        model.reset_metrics()

        # Training data
        while True:
            anchors, positives, negatives = train_data.next_triplet(k=BATCH_SIZE,
                                                                    num_points=args.num_points,
                                                                    augmentation=train_augmentations)
            
            point_cloud = tf.concat([anchors, positives, negatives], axis=0, name="point_cloud")

            if anchors is None or anchors.shape[0] != BATCH_SIZE:
                break
            
            # visualise input data
            # print("> Type of anchor input:", type(anchors))
            # print("> Type of point_cloud input:", type(point_cloud))
            # # print("> Type of next_triplet input:", type(next_triplet))
            # print("> Shape of anchor:", anchors.shape)    # 6, 4096, 6 for now
            # print("> Shape of point_cloud:", point_cloud.shape)    # 6, 4096, 6 for now
            # print("> Shape of next_triplet:", next_triplet.shape)    # 6, 4096, 6 for now

            # Training
            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            with tf.GradientTape(persistent=False, watch_accessed_variables=False) as tape_train:
                tape_train.watch([model.trainable_weights])

                # Run forward pass
                _1, features, att, _3 = model(point_cloud, training=True)

                loss_val = loss_fn(att, features)
            
            grads = tape_train.gradient(loss_val, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            
            logger.info("Loss at epoch {} step {}: {}".format(iEpoch, step, loss_val.numpy()))

            with train_writer.as_default():
                tf.summary.scalar("Loss", loss_val, step=step)

            if step % args.checkpoint_every_n_steps == 0:
                checkpoint_path = os.path.join(checkpoint_dir, "ckpt_{}".format(step))
                model.save_weights(checkpoint_path)
                logger.info("At step {}, saved checkpoint at {}.".format(step, checkpoint_path))

            savedModel_every_n_steps = 5000
            if step % savedModel_every_n_steps == 0:
                model.save(model_savepath)
                logger.info("At step {}, saved SavedModel at {}.".format(step, model_savepath))
            
            # # Run through validation data
            if step % args.validate_every_n_steps == 0 or step == 1:
                # print()
                # ---------------------------- TEST EVAL -----------------------
                fp_rate = validate(model, val_folder, val_groundtruths, args.data_dim)

                with test_writer.as_default():
                    tf.summary.scalar("fp_rate", fp_rate, step=step)

                logger.info('Validation for step %i. FP Rate: %f', step, fp_rate)
                # ---------------------------- TEST EVAL End -----------------------

                test_writer.flush()
                train_writer.flush()

            step += 1
            # print()
    
    model.save(model_savepath)
    logger.info("Before finishing training at epoch {} and step {},\
        saved model at {}.".format(args.num_epochs-1, step, model_savepath))

def load_validation_groundtruths(fname, proportion=1):
    groundtruths = []
    iGt = 0
    with open(fname) as fid:
        fid.readline()
        for line in fid:
            groundtruths.append((iGt, int(line.split()[-1])))
            iGt += 1

    if 0 < proportion < 1:
        skip = int(1.0/proportion)
        groundtruths = groundtruths[0::skip]

    return groundtruths

def validate(model, val_folder, val_groundtruths, data_dim):
    
    if val_groundtruths is None or len(val_groundtruths)==0:
        return 1

    positive_dist = []
    negative_dist = []

    for iTest in range(0, len(val_groundtruths), NUM_CLUSTERS):
        
        clouds1, clouds2 = [], []
        # We batch the validation by stacking all the validation clusters into a single point cloud,
        # while keeping them apart such that they do not overlap each other. This way NUM_CLUSTERS
        # clusters can be computed in a single pass
        for jTest in range(iTest, min(iTest + NUM_CLUSTERS, len(val_groundtruths))):
            offset = (jTest - iTest) * 100
            cluster_idx = val_groundtruths[jTest][0]

            cloud1 = DataGenerator.load_point_cloud(
                os.path.join(val_folder, '{}_0.bin'.format(cluster_idx)), data_dim)
            cloud1[:, 0] += offset
            clouds1.append(cloud1)

            cloud2 = DataGenerator.load_point_cloud(
                os.path.join(val_folder, '{}_1.bin'.format(cluster_idx)), data_dim)
            cloud2[:, 0] += offset
            clouds2.append(cloud2)

        offsets = np.arange(0, NUM_CLUSTERS * 100, 100)
        num_clusters = min(len(val_groundtruths) - iTest, NUM_CLUSTERS)
        offsets[num_clusters:] = 0
        offsets = np.pad(offsets[:, None], ((0, 0), (0, 2)), mode='constant', constant_values=0)[None, :, :]

        clouds1 = np.concatenate(clouds1, axis=0)[None, :, :]
        clouds2 = np.concatenate(clouds2, axis=0)[None, :, :]

        xyz1, features1, _, _ = model(inputs=clouds1, training=False)

        xyz2, features2, _, _ = model(inputs=clouds2, training=False)

        d = np.sqrt(np.sum(np.square(np.squeeze(features1 - features2)), axis=1))
        d = d[:num_clusters]

        positive_dist += [d[i] for i in range(len(d)) if val_groundtruths[iTest + i][1] == 1]
        negative_dist += [d[i] for i in range(len(d)) if val_groundtruths[iTest + i][1] == 0]

    d_at_95_recall = np.percentile(positive_dist, 95)
    num_FP = np.count_nonzero(np.array(negative_dist) < d_at_95_recall)
    num_TN = len(negative_dist) - num_FP
    fp_rate = num_FP / (num_FP + num_TN)

    return fp_rate

if __name__ == '__main__':

    gpu_list = []
    for i in args.gpu:
        gpu_list.append("/gpu:" + i)

    print("Intending to run using GPUS: {}".format([i for i in gpu_list]))

    # Set GPU growth
    # # print("Selecting GPU", args.gpu)
    # gpus = tf.config.list_physical_devices('GPU')
    # tf.config.set_visible_devices(gpus[args.gpu], 'GPU')
    # gpu_string = '/gpu:{}'.format(args.gpu)

    # gpu_dev = gpus[ int(args.gpu) ]

    # tf.config.experimental.set_memory_growth(gpu_dev, True)

    train(gpu_list)