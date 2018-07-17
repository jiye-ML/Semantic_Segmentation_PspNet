import tensorflow as tf
import os
import time
from scipy import misc
import numpy as np
import tensorflow.contrib.metrics as tcm

from PspNet import PSPNet101, PSPNet50
from Tools import Tools, ImageTools, IMG_MEAN


class PspNetRunner:

    def __init__(self, sess, data, config):
        # 会话
        self.sess = sess
        self.coord = None
        self.conf = config
        # 数据
        self.images_tensor, self.labels_tensor = data.get_next_data()
        # 保存
        self.saver = None
        # 训练
        self.train_op = None
        self.reduced_loss = None
        # 学习
        self.learning_rate_tensor = None

        # 网络
        self.net = PSPNet50(self.conf)

        pass

    # train
    def train(self):

        self.train_setup()

        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


        self.saver = tf.train.Saver(tf.global_variables())

        # Load the pre-trained model if provided
        # 模型加载
        ckpt = tf.train.get_checkpoint_state(self.conf.checkpoints_path)
        load_step = 0
        decay_rate = self.conf.decay_rate
        if ckpt and ckpt.model_checkpoint_path:
            print('continue training from previous checkpoint')
            load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
            # 加载衰减值
            decay_rate = pow(self.conf.decay_rate, load_step // self.conf.decay_steps)
            # 加载学习率
            self.sess.run(tf.assign(self.learning_rate_tensor, self.conf.learning_rate * decay_rate))
            self.load(ckpt_path=ckpt.model_checkpoint_path)
        else:
            Tools.new_dir(self.conf.checkpoints_path)

        # Start queue threads.
        threads = tf.train.start_queue_runners(coord=self.coord, sess=self.sess)
        # Train!
        for step in range(load_step, self.conf.num_steps + 1):

            # 学习率衰减
            if step != 0 and step % self.conf.decay_steps == 0:
                self.sess.run(tf.assign(self.learning_rate_tensor,
                                        self.sess.run(self.learning_rate_tensor) * decay_rate))
            start_time = time.time()
            if step % self.conf.save_pred_every == 0:
                summary, loss_value, _, lr = self.sess.run([self.total_summary, self.reduced_loss,
                                                            self.train_op, self.learning_rate_tensor],)
                self.save(self.conf.checkpoints_path, step)
                self.summary_writer.add_summary(summary, step)
            else:
                loss_value, _, lr = self.sess.run([self.reduced_loss, self.train_op, self.learning_rate_tensor])
            duration = time.time() - start_time
            print('step {:5d} \t lr = {:.10f} loss = {:.3f}, ({:.3f} sec/step)'.format(step, lr, loss_value, duration))

        # finish
        self.coord.request_stop()
        self.coord.join(threads)
        pass

    def train_setup(self):
        tf.set_random_seed(self.conf.random_seed)

        # Create queue coordinator.
        self.coord = tf.train.Coordinator()

        # 改变数据大小
        shape = tf.shape(self.images_tensor)
        h, w = (tf.maximum(self.conf.input_size, shape[1]), tf.maximum(self.conf.input_size, shape[2]))
        img = tf.image.resize_nearest_neighbor(self.images_tensor, [h, w])
        raw_output = self.net.fit({'data': img})

        # According from the prototxt in Caffe implement, learning rate must multiply by 10.0 in pyramid module
        fc_list = ['conv5_3_pool1_conv', 'conv5_3_pool2_conv', 'conv5_3_pool3_conv', 'conv5_3_pool6_conv', 'conv6',
                   'conv5_4']
        # 是否训练 batch_norm 的 beta 和gamma
        all_trainable = [v for v in tf.trainable_variables() if
                         ('beta' not in v.name and 'gamma' not in v.name) or self.conf.train_beta_gamma]
        fc_trainable = [v for v in all_trainable if v.name.split('/')[0] in fc_list]
        conv_trainable = [v for v in all_trainable if v.name.split('/')[0] not in fc_list]  # lr * 1.0
        fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name]  # lr * 10.0
        fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name]  # lr * 20.0
        assert (len(all_trainable) == len(fc_trainable) + len(conv_trainable))
        assert (len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))

        # Predictions: 忽略所有label >= n_classes 和 ignore的label
        raw_prediction = tf.reshape(raw_output, [-1, self.conf.num_classes])
        label_proc = ImageTools.prepare_label(self.labels_tensor, tf.shape(raw_output)[1:3],
                                              num_classes=self.conf.num_classes, one_hot=False)  # [batch_size, h, w]
        raw_gt = tf.reshape(label_proc, [-1, ])
        indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, self.conf.num_classes - 1)), 1)
        gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
        prediction = tf.gather(raw_prediction, indices)

        # Pixel-wise softmax loss.
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
        # 权值衰减
        l2_losses = [self.conf.weight_decay * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
        # loss
        self.reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)

        # 学习
        self.learning_rate_tensor = tf.Variable(self.conf.learning_rate,
                                                trainable=False, collections=[tf.GraphKeys.LOCAL_VARIABLES])

        # 滑动平均
        # save moving average
        update_ops = None
        if self.conf.update_mean_var == True:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            opt_conv = tf.train.MomentumOptimizer(self.learning_rate_tensor, self.conf.momentum)
            opt_fc_w = tf.train.MomentumOptimizer(self.learning_rate_tensor * 10.0, self.conf.momentum)
            opt_fc_b = tf.train.MomentumOptimizer(self.learning_rate_tensor * 20.0, self.conf.momentum)
            # 2018-07-16 修改优化方法
            # opt_conv = tf.train.AdamOptimizer(self.learning_rate_tensor)
            # opt_fc_w = tf.train.AdamOptimizer(self.learning_rate_tensor * 10.0)
            # opt_fc_b = tf.train.AdamOptimizer(self.learning_rate_tensor * 20.0)

            grads = tf.gradients(self.reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
            grads_conv = grads[:len(conv_trainable)]
            grads_fc_w = grads[len(conv_trainable): (len(conv_trainable) + len(fc_w_trainable))]
            grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

            train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
            train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
            train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

            self.train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)

        # Training summary
        # Processed predictions: for visualisation.
        self.pred = tf.expand_dims(tf.argmax(tf.image.resize_bilinear(raw_output,
                                                                      [self.conf.input_size, self.conf.input_size]),
                                             axis=3), dim=3)
        # Image summary.
        images_summary = tf.py_func(ImageTools.inv_preprocess, [self.images_tensor, 1, IMG_MEAN], tf.uint8)
        labels_summary = ImageTools.decode_labels(self.labels_tensor, 1, self.conf.num_classes)
        preds_summary = ImageTools.decode_labels(self.pred, 1, self.conf.num_classes)
        self.total_summary = tf.summary.image('images',
                                              tf.concat(axis=2, values=[images_summary, labels_summary, preds_summary]),
                                              max_outputs=1)  # Concatenate row-wise.
        if not os.path.exists(self.conf.logdir):
            os.makedirs(self.conf.logdir)
        self.summary_writer = tf.summary.FileWriter(self.conf.logdir, graph=tf.get_default_graph())
        pass

    def test(self):

        # 图上下文
        with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

            # 改变数据大小
            shape = tf.shape(self.images_tensor)
            h, w = (tf.maximum(self.conf.input_size, shape[1]), tf.maximum(self.conf.input_size, shape[2]))
            img = tf.image.resize_nearest_neighbor(self.images_tensor, [h, w])

            # logits
            logits = self.net.fit({"data": img})
            # 预测
            raw_output_up = tf.image.resize_bilinear(logits, size=[h, w], align_corners=True)
            raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, shape[1], shape[2])
            raw_output_up = tf.argmax(raw_output_up, dimension=3)
            prediction = tf.expand_dims(raw_output_up, dim=3)

            # 评估
            pred = tf.reshape(prediction, [-1, ])
            gt = tf.reshape(self.labels_tensor, [-1, ])
            temp = tf.less_equal(gt, self.conf.num_classes - 1)
            weights = tf.cast(temp, tf.int32)
            gt = tf.where(temp, gt, tf.cast(temp, tf.uint8))
            acc, acc_update_op = tcm.streaming_accuracy(pred, gt, weights=weights)
            # confusion matrix
            confusion_matrix_tensor = tcm.confusion_matrix(pred, gt, num_classes=self.conf.num_classes, weights=weights)

            # 启动初始化
            sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
            # 保存
            saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=100)

            # 模型加载
            ckpt = tf.train.get_checkpoint_state(self.conf.checkpoints_path)
            if ckpt and ckpt.model_checkpoint_path:
                print('test from {}'.format(ckpt.model_checkpoint_path))
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ("请先训练模型..., Train.py first")

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord, sess=sess)

            # test
            confusion_matrix = np.zeros((self.conf.num_classes, self.conf.num_classes), dtype=np.int)
            for i in range(self.conf.test_num_steps // self.conf.batch_size):
                start = time.time()
                pred, _,  c_matrix = sess.run([prediction, acc_update_op, confusion_matrix_tensor])
                confusion_matrix += c_matrix
                _diff_time = time.time() - start
                print('{}: cost {:.0f}ms'.format(i, _diff_time * 1000))
            # 总体
            self.compute_IoU_per_class(confusion_matrix)
            print("Pascal VOC 2012 validation dataset pixel accuracy: " + str(sess.run(acc)))

            coord.request_stop()
            coord.join(threads)
        pass

    # prediction
    def predict(self):

        # Create directory
        Tools.new_dir(self.conf.output_dir)

        input_images = tf.placeholder(tf.float32, shape=[None, None, 3], name='input_images')

        # preprocess images
        img_shape = tf.shape(input_images)
        h, w = (tf.maximum(self.conf.input_size, img_shape[0]), tf.maximum(self.conf.input_size, img_shape[1]))
        img = ImageTools.preprocess(input_images, h, w)
        logits = self.net.fit({'data': img})

        # Predictions.
        raw_output_up = tf.image.resize_bilinear(logits, size=[h, w], align_corners=True)
        raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
        raw_output_up = tf.expand_dims(tf.argmax(raw_output_up, axis=3), 3)
        pred = ImageTools.decode_labels(raw_output_up, 1, self.conf.num_classes)
        # pred = tf.py_func(ImageTools.decode_labels3, [raw_output_up, 1, self.conf.num_classes], tf.uint8)

        self.sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        # 初始化
        self.saver = tf.train.Saver(tf.global_variables())
        # 模型加载
        ckpt_state = tf.train.get_checkpoint_state(self.conf.checkpoints_path)
        model_path = os.path.join(self.conf.checkpoints_path, os.path.basename(ckpt_state.model_checkpoint_path))
        print('Restore from {}'.format(model_path))
        self.load(model_path)

        # 获得目录下的所有图片
        im_fn_list = self._get_images()
        for im_fn in im_fn_list:
            start = time.time()
            # 获得一张图片
            image_tensor, file_name = ImageTools.load_img(im_fn)
            image = self.sess.run(image_tensor)
            # 可视化
            mask = self.sess.run(pred, feed_dict = {input_images: image})
            # 保存
            misc.imsave("{}/{}.png".format(self.conf.output_dir, file_name), image)
            misc.imsave("{}/{}_mask.png".format(self.conf.output_dir, file_name), mask[0])
            _diff_time = time.time() - start
            print('{}: cost {:.0f}ms'.format(im_fn, _diff_time * 1000))
        pass

    # predict的时候，从文件夹获得图片
    def _get_images(self):
        files = []
        exts = ['jpg', 'png', 'jpeg', 'JPG']
        for parent, dirnames, filenames in os.walk(self.conf.predict_data_path):
            for filename in filenames:
                for ext in exts:
                    if filename.endswith(ext):
                        files.append(os.path.join(parent, filename))
                        break
        print('Find {} images'.format(len(files)))
        return files

    # 模型保存
    def save(self, ckpt_path, step):
        checkpoint_path = os.path.join(ckpt_path, 'model.ckpt')

        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        self.saver.save(self.sess, checkpoint_path, global_step=step)
        print('The checkpoint has been created.')
        pass

    # 模型加载
    def load(self, ckpt_path):
        self.saver.restore(self.sess, ckpt_path)
        print("Restored model parameters from {}".format(ckpt_path))
        pass

    # 每一类的iou和 miou
    def compute_IoU_per_class(self, confusion_matrix):
        mIoU = 0
        for i in range(self.conf.num_classes):
            # IoU = true_positive / (true_positive + false_positive + false_negative)
            TP = confusion_matrix[i, i]
            FP = np.sum(confusion_matrix[:, i]) - TP
            FN = np.sum(confusion_matrix[i]) - TP
            IoU = TP / (TP + FP + FN)
            print('class {}: {}'.format(i, IoU))
            mIoU += IoU / self.conf.num_classes
        print('mIoU: %.3f' % mIoU)
        pass

    pass