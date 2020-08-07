# Copyright 2020 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.special import expit

from bert import modeling
from bert import tokenization

import multilabel

flags = tf.flags
tfd = tfp.distributions
FLAGS = flags.FLAGS

## Optional parameters

flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "init_checkpoint",
    "gs://kats-uncertainty-estimation/multi_cased_L-12_H-768_A-12",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 512, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 512, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 512, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 100,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("save_summary_steps", 20,
                     "Save summaries every this many steps.")

flags.DEFINE_integer(
    "keep_checkpoint_max", 0,
    "The maximum number of recent checkpoint files to keep."
    "If None or 0, all checkpoint files are kept.")

flags.DEFINE_integer(
    "log_step_count_steps", 20,
    "The frequency, in number of global steps, that the global step"
    " and the loss will be logged during training. ")

flags.DEFINE_integer("iterations_per_loop", 20,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("freeze_bert", False,
                  "Whether to freeze BERT layers during fine-tuning.")

flags.DEFINE_string(
    "finetune_module", "original",
    "The module used in fine tuning for multilabel classification. "
    "The default is the original approach in BERT.")

flags.DEFINE_bool(
    "convert_tsv_to_tfrecord", False,
    "Whether to convert tsv files to tfrecord before building input_fn.")

flags.DEFINE_integer("sample_size", 1,
                     "Number of samples to collect during prediction.")

flags.DEFINE_list("label_mask", [1] * len(multilabel.LABEL_COLUMN),
                  "Whether to consider specific labels in loss.")


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True."
        )

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.io.gfile.makedirs(FLAGS.output_dir)
    tf.logging.info("***** FLAGS *****")
    writer = tf.io.gfile.GFile(f"{FLAGS.output_dir}/flags.txt", "w+")
    for key, val in FLAGS.__flags.items():
        tf.logging.info("  %s = %s", key, str(val.value))
        writer.write("%s = %s\n" % (key, str(val.value)))
    writer.close()

    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file,
                                           do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host),
        model_dir=FLAGS.output_dir,
        tf_random_seed=100,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        log_step_count_steps=FLAGS.log_step_count_steps)

    train_task_name = FLAGS.task_name if FLAGS.task_name else "train"
    _, num_train_examples, _, num_train_steps = multilabel.prepare_tfrecords(
        train_task_name, FLAGS.data_dir, FLAGS.max_seq_length, tokenizer, False,
        FLAGS.train_batch_size, 1, FLAGS.convert_tsv_to_tfrecord)
    num_train_steps = int(num_train_steps * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = multilabel.model_fn_builder(
        bert_config=bert_config,
        num_labels=len(multilabel.LABEL_COLUMN),
        init_checkpoint=tf.train.latest_checkpoint(FLAGS.init_checkpoint),
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu,
        freeze_bert=FLAGS.freeze_bert,
        finetune_module=FLAGS.finetune_module,
        num_train_examples=num_train_examples)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train and FLAGS.do_eval and (not FLAGS.use_tpu):
        eval_task_name = "eval"
        _, num_eval_examples, num_padded_eval_examples, num_eval_steps = multilabel.prepare_tfrecords(
            eval_task_name, FLAGS.data_dir, FLAGS.max_seq_length, tokenizer,
            FLAGS.use_tpu, FLAGS.eval_batch_size, 1,
            FLAGS.convert_tsv_to_tfrecord)

        tf.logging.info("***** Running training and evaluation *****")
        tf.logging.info("  Num training examples = %d", num_train_examples)
        tf.logging.info("  Training batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num training steps = %d", num_train_steps)
        tf.logging.info("  Num eval examples = %d ", num_eval_examples)
        tf.logging.info("  Eval batch size = %d", FLAGS.eval_batch_size)
        tf.logging.info("  Num eval steps = %d", num_eval_steps)

        train_input_fn = multilabel.file_based_input_fn_builder(
            input_file=f"{FLAGS.data_dir}/{train_task_name}-1.tfrecord",
            seq_length=FLAGS.max_seq_length,
            is_training=True,
            drop_remainder=True)
        eval_input_fn = multilabel.file_based_input_fn_builder(
            input_file=f"{FLAGS.data_dir}/{eval_task_name}-1.tfrecord",
            seq_length=FLAGS.max_seq_length,
            is_training=False,
            drop_remainder=False)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                            max_steps=num_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                          steps=num_eval_steps,
                                          start_delay_secs=60,
                                          throttle_secs=120)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    else:
        if FLAGS.do_train:
            tf.logging.info("***** Running training *****")
            tf.logging.info("  Num examples = %d", num_train_examples)
            tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
            tf.logging.info("  Num steps = %d", num_train_steps)
            train_input_fn = multilabel.file_based_input_fn_builder(
                input_file=f"{FLAGS.data_dir}/{train_task_name}-1.tfrecord",
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True)
            estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

        if FLAGS.do_eval:
            eval_task_name = "eval"
            ckpt_steps = list(
                range(0, num_train_steps, FLAGS.save_checkpoints_steps))
            ckpt_steps.append(num_train_steps)
            best_result = multilabel.eval_routine(
                eval_task_name, FLAGS.data_dir, FLAGS.output_dir,
                FLAGS.max_seq_length, tokenizer, estimator, FLAGS.use_tpu,
                FLAGS.predict_batch_size, ckpt_steps, 1,
                FLAGS.convert_tsv_to_tfrecord)

    if FLAGS.do_predict:
        predict_task_name = "predict"
        output = multilabel.predict_routine(
            predict_task_name, FLAGS.data_dir, FLAGS.output_dir,
            FLAGS.max_seq_length, tokenizer, estimator, FLAGS.use_tpu,
            FLAGS.predict_batch_size, FLAGS.sample_size,
            FLAGS.convert_tsv_to_tfrecord)
        file_name = f"{FLAGS.output_dir}/{predict_task_name}-{FLAGS.sample_size}-results.tsv"
        with tf.io.gfile.GFile(file_name, "w+") as writer:
            num_written_items = 0
            writer.write(
                multilabel.ID_COLUMN + "\t" + multilabel.LANG_COLUMN + "\t" +
                "\t".join(name + " prob" for name in multilabel.LABEL_COLUMN) +
                "\t" +
                "\t".join(name + " var" for name in multilabel.LABEL_COLUMN) +
                "\t" +
                "\t".join(name + " ci_lb" for name in multilabel.LABEL_COLUMN) +
                "\t" +
                "\t".join(name + " ci_ub" for name in multilabel.LABEL_COLUMN) +
                "\t" + "\t".join(
                    name + " ci_68_lb" for name in multilabel.LABEL_COLUMN) +
                "\t" + "\t".join(name + " ci_68_ub"
                                 for name in multilabel.LABEL_COLUMN) + "\n")
            for (i, pred) in enumerate(output):
                logits = np.array(pred["logits"])
                vars = np.array(pred["variance"])
                std_dev = np.sqrt(vars)
                lower_95_ci = expit(logits - 2 * std_dev)
                upper_95_ci = expit(logits + 2 * std_dev)
                lower_68_ci = expit(logits - std_dev)
                upper_68_ci = expit(logits + std_dev)
                output_line = pred[multilabel.ID_COLUMN] + "\t" + pred[
                    multilabel.LANG_COLUMN] + "\t"
                output_line += "\t".join(
                    str(class_prob) for class_prob in pred["probs"]) + "\t"
                output_line += "\t".join(
                    str(class_var) for class_var in vars) + "\t"
                output_line += "\t".join(str(lb) for lb in lower_95_ci) + "\t"
                output_line += "\t".join(str(ub) for ub in upper_95_ci) + "\t"
                output_line += "\t".join(str(lb) for lb in lower_68_ci) + "\t"
                output_line += "\t".join(str(ub) for ub in upper_68_ci) + "\n"
                writer.write(output_line)
                num_written_items += 1
            assert num_written_items == len(output)
        tf.logging.info(f"Prediction results written to {file_name}")


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
