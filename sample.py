"""Sample from a pre-trained IAF model.

Example:
 python sample.py --logdir=/home/craffel/iaf/log/ \
   --n_samples 10000 \
   --hpconfig depth=1,num_blocks=20,kl_min=0.1,learning_rate=0.002,batch_size=32,eval_batch_size=100 \  # noqa
   --num_gpus 1
"""

import numpy as np
import tensorflow as tf

from tf_utils.common import CheckpointLoader
from tf_train import CVAE1
import tf_train

flags = tf.flags
flags.DEFINE_integer("n_samples", 10000, "Total number of samples to make.")
flags.DEFINE_string("sample_path", "samples.npz", "Path to .npz sample file.")
FLAGS = flags.FLAGS


def sample(hps):
    tf.reset_default_graph()

    with tf.variable_scope("model") as vs:
        model = CVAE1(hps, "eval")
        vs.reuse_variables()
        sample_model = CVAE1(hps, "sample")

    saver = tf.train.Saver(model.avg_dict)
    # Use only 4 threads for the evaluation.
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=4,
                            inter_op_parallelism_threads=4)
    sess = tf.Session(config=config)
    ckpt_loader = CheckpointLoader(
        saver, model.global_step, FLAGS.logdir + "/train")

    with sess.as_default():
        while ckpt_loader.load_checkpoint():
            global_step = ckpt_loader.last_global_step
            tf.logging.info("Loaded ckpt, global step {}".format(global_step))
            break
        n_sampled = 0
        samples = np.zeros((FLAGS.n_samples, 32, 32, 3), np.float32)
        while n_sampled < FLAGS.n_samples:
            # Generate samples
            sample_x = sess.run(sample_model.m_trunc[0])
            # Convert to BHWC
            sample_x = np.transpose(sample_x, [0, 2, 3, 1])
            # Move to range [0, 1]
            sample_x += .5
            # Store samples
            samples[n_sampled:n_sampled + sample_x.shape[0]] = sample_x
            n_sampled += sample_x.shape[0]
            tf.logging.info("Sampled {} images, {} total of {}".format(
                sample_x.shape[0], n_sampled, FLAGS.n_samples))
        np.savez_compressed(FLAGS.sample_path, samples=samples)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    hps = tf_train.get_default_hparams().parse(FLAGS.hpconfig)
    hps.batch_size = hps.eval_batch_size
    hps.num_gpus = 1
    sample(hps)
