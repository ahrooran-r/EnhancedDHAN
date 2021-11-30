import glob
import pathlib

from networks import *
from utils import *

EPS = 1e-12


class Deshadower(object):

    def __init__(self, model_path, vgg_19_path, use_gpu):
        self.vgg_19_path = vgg_19_path
        self.model = model_path
        self.channel = 64
        if use_gpu < 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(use_gpu)
        self.__setup_model()

    def __setup_model(self):

        with tf.variable_scope(tf.get_variable_scope()):
            input = tf.placeholder(tf.float32, shape=[None, None, None, 3])
            self.shadow_free_image = build_aggasatt_joint(input, 64, self.vgg_19_path)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        idtd_ckpt = tf.train.get_checkpoint_state(self.model)
        saver_restore = tf.train.Saver([var for var in tf.trainable_variables()])
        print('loaded ' + idtd_ckpt.model_checkpoint_path)
        saver_restore.restore(self.sess, idtd_ckpt.model_checkpoint_path)

    def run(self, source_path, target_path, width, height, ext=["jpg", "png"]):

        original_files = []
        [original_files.extend(glob.glob(source_path + '*.' + e)) for e in ext]

        print(original_files)
        # for img_path in original_files:
        #     iminput = cv2.resize(cv2.imread(img_path, -1), (width, height), interpolation=cv2.INTER_CUBIC)
        #
        #     imoutput = self.sess.run(
        #         self.shadow_free_image,
        #         feed_dict={
        #             input: np.expand_dims(iminput / 255., axis=0)
        #         }
        #     )
        #
        #     imoutput = np.uint8(np.squeeze(np.minimum(np.maximum(imoutput[0], 0.0), 1.0)) * 255.0)
        #
        #     imname = pathlib.Path(img_path).stem
        #     cv2.imwrite(os.path.join(target_path, f'{imname}.jpg'), imoutput)
        #     cv2.waitKey(0)
        #
        #     print(f"completed {imname}.jpg")
