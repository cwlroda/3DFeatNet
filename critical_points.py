import tensorflow as tf
import numpy as np
import sys, os

sys.path.append(os.path.dirname('../'))

from data.datagenerator import DataGenerator

def tf_compute_critical_points(gradient, pointclouds, number_of_keypoints, drop_negative=False, power=1):
    end_points = {}

    xyz = pointclouds[:, :, :3]
    end_points['xyz'] = xyz

    # detect keypoints
    sphere_core = tf.contrib.distributions.percentile(xyz, 50.0, interpolation='lower', axis=1, keep_dims=True)
    sphere_core += tf.contrib.distributions.percentile(xyz, 50.0, interpolation='higher', axis=1, keep_dims=True)
    sphere_core /= 2.
    end_points['sphere_core'] = sphere_core

    sphere_r = tf.sqrt(tf.reduce_sum(tf.square(xyz - sphere_core), axis=2))  ## BxN
    end_points['sphere_r'] = sphere_r

    sphere_axis = xyz - sphere_core  ## BxNx3
    end_points['sphere_axis'] = sphere_axis

    if drop_negative:
        sphere_map = tf.multiply(tf.reduce_sum(tf.multiply(gradient, sphere_axis), axis=2),
                                 tf.pow(sphere_r, power))
    else:
        sphere_map = -tf.multiply(tf.reduce_sum(tf.multiply(gradient, sphere_axis), axis=2),
                                  tf.pow(sphere_r, power))

    end_points['sphere_map'] = sphere_map

    drop_indice = tf.nn.top_k(sphere_map, k=number_of_keypoints)[1]
    end_points['drop_indice'] = drop_indice

    batch_size = tf.shape(xyz)[0]
    ranges = tf.range(batch_size, dtype=tf.int32)  # [0,1,...B]
    ranges = tf.reshape(ranges, [-1, 1])
    ranges_2d = tf.tile(ranges, [1, number_of_keypoints])
    ranges_2d_reshaped = tf.reshape(ranges_2d, [-1, 1])# [0,0,0,1,1,1....B*K]

    end_points['ranges'] = ranges
    end_points['ranges_2d'] = ranges_2d
    end_points['ranges_2d_reshaped'] = ranges_2d_reshaped

    drop_indice_reshaped = tf.reshape(drop_indice, [-1, 1])  #(B*K),1
    end_points['drop_indice_reshaped'] = drop_indice_reshaped

    drop_indice_2d_mm = tf.concat((ranges_2d_reshaped, drop_indice_reshaped), axis=1)  # BxK
    end_points['drop_indice_2d_mm'] = drop_indice_2d_mm

    xyz_kp = tf.gather_nd(xyz, drop_indice_2d_mm)  # (B*K)x3
    xyz_kp = tf.reshape(xyz_kp, (batch_size, number_of_keypoints, 3))  # BxKx3
    end_points['xyz_kp'] = xyz_kp

    return xyz_kp, end_points

def detect_critical_points(gradient, pointclouds, number_of_keypoints, drop_negative=False, power=1):
    end_points = {}

    xyz = pointclouds[:, :, :3]
    end_points['xyz'] = xyz

    # detect keypoints
    sphere_core = np.median(xyz, axis=1, keepdims=True)
    end_points['sphere_core'] = sphere_core

    sphere_r = np.sqrt(np.sum(np.square(xyz - sphere_core), axis=2))  ## BxN
    end_points['sphere_r'] = sphere_r

    sphere_axis = xyz - sphere_core  ## BxNx3
    end_points['sphere_axis'] = sphere_axis

    if drop_negative:
        sphere_map = np.multiply(np.sum(np.multiply(gradient, sphere_axis), axis=2),
                                 np.power(sphere_r, power))
    else:
        sphere_map = -np.multiply(np.sum(np.multiply(gradient, sphere_axis), axis=2),
                                  np.power(sphere_r, power))

    end_points['sphere_map'] = sphere_map

    drop_indice = np.argpartition(sphere_map, kth=sphere_map.shape[1] - number_of_keypoints, axis=1)[:, -number_of_keypoints:]
    end_points['drop_indice'] = drop_indice

    xyz_kp = np.zeros((xyz.shape[0], number_of_keypoints, 3), dtype=float)
    for j in range(xyz.shape[0]):
        xyz_kp[j, :, :] = xyz[j, drop_indice[j], :3]
    end_points['xyz_kp'] = xyz_kp

    return xyz_kp, end_points


def checkEqual(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)

if __name__ == "__main__":
    print("Testing pc saliency implementation...")

    number_of_keypoints = 512

    pointcloud = DataGenerator.load_point_cloud('/home/gtinchev/code/SalientKeypointComparator/example_data/test/pointclouds.bin', num_cols=6)
    gradients = DataGenerator.load_point_cloud('/home/gtinchev/code/SalientKeypointComparator/example_data/test/gradients.bin', num_cols=3)
    pointcloud2 = DataGenerator.load_point_cloud('/home/gtinchev/code/SalientKeypointComparator/example_data/test/pointclouds2.bin', num_cols=6)
    gradients2 = DataGenerator.load_point_cloud('/home/gtinchev/code/SalientKeypointComparator/example_data/test/gradients2.bin', num_cols=3)

    # extend this to have a batch dimension
    p_batches = pointcloud[np.newaxis, :]
    g_batches = gradients[np.newaxis, :]
    p2_batches = pointcloud2[np.newaxis, :]
    g2_batches = gradients2[np.newaxis, :]

    pointcloud_batches = np.concatenate((p_batches, p2_batches), axis=0)
    gradients_batches = np.concatenate((g_batches, g2_batches), axis=0)

    assert np.shape(pointcloud_batches) == (2, 16384, 6), "Invalid pointcloud shape"
    assert np.shape(gradients_batches) == (2, 16384, 3), "Invalid gradients shape"

    # Extract critical points:
    crit_points, end_points_np = detect_critical_points(gradients_batches, pointcloud_batches, number_of_keypoints)

    assert np.shape(crit_points) == (2, number_of_keypoints, 3), "Invalid critical points shape"

    # now do the same with tensorflow
    pc_pl = tf.placeholder(tf.float32, shape=(None, None, 6))
    gr_pl = tf.placeholder(tf.float32, shape=(None, None, 3))

    # Ops
    crit_points_op, end_points_op = tf_compute_critical_points(gr_pl, pc_pl, number_of_keypoints)

    with tf.Session() as sess:
        crit_points_tf, end_points_tf = sess.run([crit_points_op, end_points_op], feed_dict={
            pc_pl: pointcloud_batches,
            gr_pl: gradients_batches
        })

    np.testing.assert_array_equal(end_points_np['xyz'], end_points_tf['xyz'], err_msg='Wrong XYZ in endpoints')
    np.testing.assert_array_equal(end_points_np['sphere_core'], end_points_tf['sphere_core'], err_msg='Wrong sphere_core in endpoints')
    np.testing.assert_array_equal(end_points_np['sphere_r'], end_points_tf['sphere_r'], err_msg='Wrong sphere_r in endpoints')
    np.testing.assert_array_equal(end_points_np['sphere_axis'], end_points_tf['sphere_axis'], err_msg='Wrong sphere_axis in endpoints')
    np.testing.assert_allclose(end_points_np['sphere_map'], end_points_tf['sphere_map'], rtol=1e-8, atol=0.05, err_msg='Wrong sphere_map in endpoints')
    np.testing.assert_array_equal(np.sort(end_points_np['drop_indice']), np.sort(end_points_tf['drop_indice']), err_msg='Wrong drop_indice in endpoints')
    np.testing.assert_allclose(np.sort(end_points_np['xyz_kp'], axis=None), np.sort(end_points_tf['xyz_kp'], axis=None), err_msg='Wrong xyz_kp in endpoints')
    assert np.shape(crit_points_tf) == np.shape(crit_points), "Keypoint shape mismatch"

    print("Successfully tested that TF implementation is equal to the numpy implementation.")
