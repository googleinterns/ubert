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

import math
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

flags.DEFINE_integer("train_batch_size", 512, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 512, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 512, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 2e-5, "The initial learning rate for Adam.")

flags.DEFINE_float(
    "num_init_train_epochs", None,
    "Total number of training epochs to perform during initial training.")

flags.DEFINE_float("num_query_train_epochs", None,
                   "Total number of training epochs to perform in each query.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 100,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("save_summary_steps", 25,
                     "Save summaries every this many steps.")

flags.DEFINE_integer(
    "keep_checkpoint_max", 0,
    "The maximum number of recent checkpoint files to keep."
    "If None or 0, all checkpoint files are kept.")

flags.DEFINE_integer(
    "log_step_count_steps", 25,
    "The frequency, in number of global steps, that the global step"
    " and the loss will be logged during training. ")

flags.DEFINE_integer("iterations_per_loop", 25,
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
    "finetune_module", None,
    "The module used in fine tuning for multilabel classification. "
    "The default is the original approach in BERT.")

flags.DEFINE_bool(
    "convert_tsv_to_tfrecord", False,
    "Whether to convert tsv files to tfrecord before building input_fn.")

flags.DEFINE_integer("n_queries", 1,
                     "Number of samples to collect during prediction.")

flags.DEFINE_integer("sample_size", 10,
                     "Number of samples to collect during prediction.")

flags.DEFINE_integer("n_instances", 1000,
                     "Number of instances to sample in each query.")

flags.DEFINE_bool(
    "retrain_all", False,
    "Whether to convert tsv files to tfrecord before building input_fn.")

flags.DEFINE_string(
    "al_query_strategy", None,
    "The module used in fine tuning for multilabel classification. "
    "The default is the original approach in BERT.")

flags.DEFINE_list("new_types", [
    "/film/actor", "/film/director", "/geography/river", "/sports/sports_team"
], "Whether to consider specific types in loss.")


def uniform(tokenizer, estimator, data_dir, model_dir, pool_name, n_instances,
            sample_size, label_distn, max_seq_length, use_tpu, batch_size,
            overwrite_tfrecord):
    del tokenizer, estimator, model_dir, sample_size, label_distn  # Unused.
    del max_seq_length, use_tpu, batch_size, overwrite_tfrecord  # Unused.
    pool_df = pd.read_csv(f"{data_dir}/{pool_name}.tsv", sep="\t")
    query_idx = np.random.choice(range(len(pool_df)),
                                 size=n_instances,
                                 replace=False)
    return pool_df[~pool_df.index.isin(query_idx)], pool_df.iloc[query_idx]


def margin_balanced(tokenizer, estimator, data_dir, model_dir, pool_name,
                    n_instances, sample_size, label_distn, max_seq_length,
                    use_tpu, batch_size, overwrite_tfrecord):
    pool_df = pd.read_csv(f"{data_dir}/{pool_name}.tsv", sep="\t")
    n_instances_per_label = [
        math.ceil(int(probs * n_instances)) for probs in label_distn
    ]
    output = multilabel.predict_routine(pool_name, data_dir, model_dir,
                                        max_seq_length, tokenizer, estimator,
                                        use_tpu, batch_size, sample_size,
                                        overwrite_tfrecord)
    result_df = pd.DataFrame(output)
    probs_df = pd.DataFrame(result_df["probs"].tolist(),
                            index=result_df.index,
                            columns=multilabel.LABEL_COLUMN)
    margin_df = np.abs(probs_df - 0.5)
    query_idx = []
    avail_idx = set(range(len(margin_df)))
    for i in range(len(multilabel.LABEL_COLUMN)):
        chosen_idx = set(margin_df[margin_df.index.isin(avail_idx)].nsmallest(
            n_instances_per_label[i], multilabel.LABEL_COLUMN[i],
            keep="all").index)
        query_idx.extend(chosen_idx)
        avail_idx -= chosen_idx
    return pool_df[~pool_df.index.isin(query_idx)], pool_df.iloc[query_idx]


def max_variance_balanced(tokenizer, estimator, data_dir, model_dir, pool_name,
                          n_instances, sample_size, label_distn, max_seq_length,
                          use_tpu, batch_size, overwrite_tfrecord):
    pool_df = pd.read_csv(f"{data_dir}/{pool_name}.tsv", sep="\t")
    n_instances_per_label = [
        math.ceil(int(probs * n_instances)) for probs in label_distn
    ]
    output = multilabel.predict_routine(pool_name, data_dir, model_dir,
                                        max_seq_length, tokenizer, estimator,
                                        use_tpu, batch_size, sample_size,
                                        overwrite_tfrecord)
    result_df = pd.DataFrame(output)
    variance_df = pd.DataFrame(result_df["variance"].tolist(),
                               index=result_df.index,
                               columns=multilabel.LABEL_COLUMN)
    query_idx = []
    avail_idx = set(range(len(variance_df)))
    for i in range(len(multilabel.LABEL_COLUMN)):
        chosen_idx = set(
            variance_df[variance_df.index.isin(avail_idx)].nlargest(
                n_instances_per_label[i],
                multilabel.LABEL_COLUMN[i],
                keep="all").index)
        query_idx.extend(chosen_idx)
        avail_idx -= chosen_idx
    return pool_df[~pool_df.index.isin(query_idx)], pool_df.iloc[query_idx]


def max_variance_pos(tokenizer, estimator, data_dir, model_dir, pool_name,
                     n_instances, sample_size, label_distn, max_seq_length,
                     use_tpu, batch_size, overwrite_tfrecord):
    pool_df = pd.read_csv(f"{data_dir}/{pool_name}.tsv", sep="\t")
    n_instances_per_label = [
        math.ceil(int(probs * n_instances)) for probs in label_distn
    ]
    output = multilabel.predict_routine(pool_name, data_dir, model_dir,
                                        max_seq_length, tokenizer, estimator,
                                        use_tpu, batch_size, sample_size,
                                        overwrite_tfrecord)
    result_df = pd.DataFrame(output)
    variance_df = pd.DataFrame(result_df["variance"].tolist(),
                               index=result_df.index,
                               columns=multilabel.LABEL_COLUMN)
    mean_probs_df = pd.DataFrame(result_df["probs"].tolist(),
                                 index=result_df.index,
                                 columns=multilabel.LABEL_COLUMN)
    query_idx = []
    avail_idx = set(range(len(variance_df)))
    for i in range(len(multilabel.LABEL_COLUMN)):
        label = multilabel.LABEL_COLUMN[i]
        pos_idx = mean_probs_df[mean_probs_df.index.isin(avail_idx) &
                                (mean_probs_df[label] > 0.5)].index
        chosen_idx = set(variance_df[variance_df.index.isin(pos_idx)].nlargest(
            n_instances_per_label[i], label, keep="all").index)
        query_idx.extend(chosen_idx)
        avail_idx -= chosen_idx
    return pool_df[~pool_df.index.isin(query_idx)], pool_df.iloc[query_idx]


def max_ub(tokenizer, estimator, data_dir, model_dir, pool_name, n_instances,
           sample_size, label_distn, max_seq_length, use_tpu, batch_size,
           overwrite_tfrecord):
    pool_df = pd.read_csv(f"{data_dir}/{pool_name}.tsv", sep="\t")
    n_instances_per_label = [
        math.ceil(int(probs * n_instances)) for probs in label_distn
    ]
    output = multilabel.predict_routine(pool_name, data_dir, model_dir,
                                        max_seq_length, tokenizer, estimator,
                                        use_tpu, batch_size, sample_size,
                                        overwrite_tfrecord)
    result_df = pd.DataFrame(output)
    variance_df = pd.DataFrame(result_df["variance"].tolist(),
                               index=result_df.index,
                               columns=multilabel.LABEL_COLUMN)
    mean_logits_df = pd.DataFrame(result_df["logits"].tolist(),
                                  index=result_df.index,
                                  columns=multilabel.LABEL_COLUMN)
    ub_68_df = expit(mean_logits_df + np.sqrt(variance_df))
    query_idx = []
    avail_idx = set(range(len(variance_df)))
    for i in range(len(multilabel.LABEL_COLUMN)):
        label = multilabel.LABEL_COLUMN[i]
        chosen_idx = set(ub_68_df[ub_68_df.index.isin(avail_idx)].nlargest(
            n_instances_per_label[i], label, keep="all").index)
        query_idx.extend(chosen_idx)
        avail_idx -= chosen_idx
    return pool_df[~pool_df.index.isin(query_idx)], pool_df.iloc[query_idx]


def get_label_mask(new_types):
    unknown_types = [int(i in new_types) for i in multilabel.LABEL_COLUMN]
    known_types = [int(not i) for i in unknown_types]
    return known_types, unknown_types


def al_teach(task_name, tokenizer, bert_config, output_dir, data_dir,
             num_train_epochs, tpu_cluster_resolver, is_per_host,
             warm_start_ckpt):
    tf.io.gfile.makedirs(output_dir)
    task_name = task_name.lower()

    run_config = tf.compat.v1.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host),
        model_dir=output_dir,
        tf_random_seed=100,
        save_summary_steps=FLAGS.save_summary_steps,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        keep_checkpoint_max=FLAGS.keep_checkpoint_max,
        log_step_count_steps=FLAGS.log_step_count_steps)

    _, num_train_examples, _, num_train_steps = multilabel.prepare_tfrecords(
        task_name, data_dir, FLAGS.max_seq_length, tokenizer, False,
        FLAGS.train_batch_size, 1, FLAGS.convert_tsv_to_tfrecord)
    num_train_steps = int(num_train_steps * num_train_epochs)
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
        predict_batch_size=FLAGS.predict_batch_size,
        warm_start_from=warm_start_ckpt)

    tf.logging.info("***** Running teaching *****")
    tf.logging.info("  Num examples = %d", num_train_examples)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = multilabel.file_based_input_fn_builder(
        input_file=f"{data_dir}/{task_name}-1.tfrecord",
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    ckpt_steps = list(range(0, num_train_steps, FLAGS.save_checkpoints_steps))
    ckpt_steps.append(num_train_steps)
    best_result = multilabel.eval_routine(FLAGS.task_name, FLAGS.data_dir,
                                          output_dir, FLAGS.max_seq_length,
                                          tokenizer, estimator, FLAGS.use_tpu,
                                          FLAGS.predict_batch_size, ckpt_steps,
                                          1, FLAGS.convert_tsv_to_tfrecord)
    return estimator, best_result


def active_learning_procedure(query_strategy, tokenizer, bert_config, data_dir,
                              output_dir, finetune_module, tpu_cluster_resolver,
                              is_per_host, max_seq_length, use_tpu, batch_size,
                              init_train_file_name, init_pool_file_name,
                              n_queries, n_instances, sample_size,
                              num_init_train_epochs, num_query_train_epochs,
                              retrain_all, overwrite_tfrecord):
    query_func = uniform
    if query_strategy == "max_variance_balanced":
        query_func = max_variance_balanced
    elif query_strategy == "margin_balanced":
        query_func = margin_balanced
    elif query_strategy == "max_variance_pos":
        query_func = max_variance_pos
    elif query_strategy == "max_ub":
        query_func = max_ub

    known_type_mask, new_type_mask = get_label_mask(FLAGS.new_types)

    estimator, result = al_teach(init_train_file_name, tokenizer, bert_config,
                                 f"{output_dir}/init", data_dir,
                                 num_init_train_epochs, tpu_cluster_resolver,
                                 is_per_host, None)
    perf_history = [result]

    init_train_df = pd.read_csv(f"{data_dir}/{init_train_file_name}.tsv",
                                sep="\t")
    label_distn = np.array(new_type_mask)
    label_distn = label_distn / sum(label_distn)
    sample_label_mask = []
    for i in range(len(label_distn)):
        mask = [0] * len(multilabel.LABEL_COLUMN)
        mask[i] = 1
        sample_label_mask.extend([str(mask)] *
                                 math.ceil(n_instances * label_distn[i]))

    for i in range(n_queries):
        if i == 0:
            model_dir = f"{output_dir}/init"
            model_data_dir = data_dir
            pool_name = init_pool_file_name
            if retrain_all:
                train_df = init_train_df
        else:
            model_dir = f"{output_dir}/{query_strategy}-{i}"
            model_data_dir = f"{data_dir}/{finetune_module}-{query_strategy}_{num_init_train_epochs}_{num_query_train_epochs}"
            pool_name = f"{num_query_train_epochs}-{i}_pool"
            if retrain_all:
                train_df = pd.read_csv(
                    f"{model_data_dir}/{num_query_train_epochs}-{i}.tsv",
                    sep="\t")

        pool_df, query_df = query_func(tokenizer, estimator, model_data_dir,
                                       model_dir, pool_name, n_instances,
                                       sample_size, label_distn, max_seq_length,
                                       use_tpu, batch_size, overwrite_tfrecord)
        query_df[multilabel.MASK_COLUMN] = sample_label_mask

        model_data_dir = f"{data_dir}/{finetune_module}-{query_strategy}_{num_init_train_epochs}_{num_query_train_epochs}"
        next_train_name = f"{num_query_train_epochs}-{i + 1}"
        next_model_dir = f"{output_dir}/{query_strategy}-{i + 1}"
        next_pool_name = f"{num_query_train_epochs}-{i + 1}_pool"

        if retrain_all:
            pd.concat([query_df, train_df
                      ]).to_csv(f"{model_data_dir}/{next_train_name}.tsv",
                                sep="\t")
            query_df.to_csv(
                f"{model_data_dir}/{next_train_name}-incremental.tsv", sep="\t")
        else:
            query_df.to_csv(f"{model_data_dir}/{next_train_name}.tsv", sep="\t")
        pool_df.to_csv(f"{model_data_dir}/{next_pool_name}.tsv",
                       sep="\t",
                       index=False)

        estimator, result = al_teach(next_train_name, tokenizer, bert_config,
                                     next_model_dir, model_data_dir,
                                     num_query_train_epochs,
                                     tpu_cluster_resolver, is_per_host,
                                     model_dir)
        perf_history.append(result)
    return perf_history


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.io.gfile.makedirs(FLAGS.output_dir)
    tf.logging.info("***** FLAGS *****")
    writer = tf.io.gfile.GFile(
        f"{FLAGS.output_dir}/{FLAGS.al_query_strategy}_flags.txt", "w+")
    for key, val in FLAGS.__flags.items():
        tf.logging.info("  %s = %s", key, str(val.value))
        writer.write("%s = %s\n" % (key, str(val.value)))
    writer.close()

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))
    tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file,
                                           do_lower_case=FLAGS.do_lower_case)
    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2

    # Active learning procedure
    results = active_learning_procedure(
        FLAGS.al_query_strategy, tokenizer, bert_config, FLAGS.data_dir,
        FLAGS.output_dir, FLAGS.finetune_module, tpu_cluster_resolver,
        is_per_host, FLAGS.max_seq_length, FLAGS.use_tpu,
        FLAGS.predict_batch_size, "train_known_type", "train_new_type",
        FLAGS.n_queries, FLAGS.n_instances, FLAGS.sample_size,
        FLAGS.num_init_train_epochs, FLAGS.num_query_train_epochs,
        FLAGS.retrain_all, FLAGS.convert_tsv_to_tfrecord)


if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    flags.mark_flag_as_required("finetune_module")
    flags.mark_flag_as_required("al_query_strategy")
    flags.mark_flag_as_required("num_init_train_epochs")
    flags.mark_flag_as_required("num_query_train_epochs")
    tf.app.run()