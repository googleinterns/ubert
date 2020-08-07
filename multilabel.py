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
"""The modified code for multilabel classification and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ast
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp

from bert import modeling
from bert import optimization
from bert import tokenization

tfd = tfp.distributions

LABEL_COLUMN = [
    "/people/person", "/location/location", "/location/citytown",
    "/biology/organism_classification", "/sports/pro_athlete",
    "/organization/organization", "/fictional_universe/fictional_character",
    "/film/actor", "/tv/tv_series_episode", "/music/artist", "/book/author",
    "/film/film", "/time/event", "/book/written_work",
    "/soccer/football_player", "/film/director", "/tv/tv_actor",
    "/tv/tv_program", "/education/educational_institution", "/geography/river",
    "/sports/sports_team", "/business/consumer_product"
]
DATA_COLUMN = "description"
ID_COLUMN = "id"
LANG_COLUMN = "language"
MASK_COLUMN = "label_mask"


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, label_mask=None):
        """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.label_mask = label_mask


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True,
                 label_mask=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id if label_id else [0] * len(LABEL_COLUMN)
        self.is_real_example = is_real_example
        self.label_mask = label_mask if label_mask else [1] * len(LABEL_COLUMN)


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([len(LABEL_COLUMN)], tf.int64),
        "is_real_example": tf.FixedLenFeature([1], tf.int64),
        "label_mask": tf.FixedLenFeature([len(LABEL_COLUMN)], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.io.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, tf.int32)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(input_ids=[0] * max_seq_length,
                             input_mask=[0] * max_seq_length,
                             segment_ids=[0] * max_seq_length,
                             label_id=[0] * len(label_list),
                             is_real_example=False,
                             label_mask=[0] * len(label_list))

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    # label_id = label_map[example.label]
    # if ex_index < 5:
    #     tf.logging.info("*** Example ***")
    #     tf.logging.info("guid: %s" % (example.guid))
    #     tf.logging.info(
    #         "tokens: %s" %
    #         " ".join([tokenization.printable_text(x) for x in tokens]))
    #     tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #     tf.logging.info("input_mask: %s" %
    #                     " ".join([str(x) for x in input_mask]))
    #     tf.logging.info("segment_ids: %s" %
    #                     " ".join([str(x) for x in segment_ids]))
    #     tf.logging.info("label_id: %s" % " ".join([str(x) for x in label_id]))

    feature = InputFeatures(input_ids=input_ids,
                            input_mask=input_mask,
                            segment_ids=segment_ids,
                            label_id=example.label,
                            is_real_example=True,
                            label_mask=example.label_mask)
    return feature


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def file_based_convert_examples_to_features(examples, label_list,
                                            max_seq_length, tokenizer,
                                            output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" %
                            (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=values))
            return f

        features = dict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_id)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])
        features["label_mask"] = create_int_feature(feature.label_mask)

        tf_example = tf.train.Example(features=tf.train.Features(
            feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings, freeze_bert,
                 finetune_module, num_train_examples, is_real_example,
                 label_mask):
    """Creates a classification model."""
    model = modeling.BertModel(config=bert_config,
                               is_training=is_training,
                               input_ids=input_ids,
                               input_mask=input_mask,
                               token_type_ids=segment_ids,
                               use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    if freeze_bert:
        output_layer = tf.stop_gradient(output_layer)

    hidden_size = output_layer.shape[-1].value
    labels = tf.cast(labels, tf.float32)

    with tf.variable_scope("loss"):
        if finetune_module == "original":
            output_weights = tf.get_variable(
                "output_weights", [num_labels, hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=0.02))

            output_bias = tf.get_variable("output_bias", [num_labels],
                                          initializer=tf.zeros_initializer())
            if is_training:
                # I.e., 0.1 dropout
                output_layer = tf.nn.dropout(output_layer, rate=0.1)

            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)

        elif finetune_module == "simple_repar":
            repar_layer = tf.keras.layers.Dense(
                num_labels * 2,
                kernel_initializer=modeling.create_initializer())(output_layer)
            logits, logvar = tf.split(repar_layer, num_or_size_splits=2, axis=1)
            eps = tf.random.normal(shape=(num_labels,))
            logits = eps * tf.exp(logvar * .5) + logits
            variance = tf.exp(logvar)

        elif finetune_module == "bnn":
            kl_divergence_function = (lambda q, p, _: tfd.kl_divergence(
                q, p) / tf.cast(num_train_examples, dtype=tf.float32))
            model = tf.keras.Sequential([
                tfp.layers.DenseFlipout(
                    hidden_size,
                    kernel_divergence_fn=kl_divergence_function,
                    activation="relu"),
                tfp.layers.DenseFlipout(
                    hidden_size,
                    kernel_divergence_fn=kl_divergence_function,
                    activation="relu"),
                tfp.layers.DenseFlipout(
                    num_labels, kernel_divergence_fn=kl_divergence_function)
            ])
            logits = model(output_layer)
            kl = tf.reduce_mean(model.losses)

        elif finetune_module == "mc":
            hidden_layer_1 = tf.keras.layers.Dense(
                hidden_size,
                activation="relu",
                kernel_initializer=modeling.create_initializer())(output_layer)
            hidden_layer_1 = tf.keras.layers.Dropout(0.25)(hidden_layer_1,
                                                           training=True)
            hidden_layer_2 = tf.keras.layers.Dense(
                hidden_size,
                activation="relu",
                kernel_initializer=modeling.create_initializer())(
                    hidden_layer_1)
            hidden_layer_2 = tf.keras.layers.Dropout(0.5)(hidden_layer_2,
                                                          training=True)
            logits = tf.keras.layers.Dense(
                num_labels, kernel_initializer=modeling.create_initializer())(
                    hidden_layer_2)

        elif finetune_module == "vae":
            encode_layer_1 = tf.keras.layers.Dense(
                hidden_size,
                activation="relu",
                kernel_initializer=modeling.create_initializer())(output_layer)
            encode_layer_2 = tf.keras.layers.Dense(
                hidden_size,
                activation="relu",
                kernel_initializer=modeling.create_initializer())(
                    encode_layer_1)
            encode_layer_3 = tf.keras.layers.Dense(
                num_labels * 2,
                kernel_initializer=modeling.create_initializer())(
                    encode_layer_2)
            logits, logvar = tf.split(encode_layer_3,
                                      num_or_size_splits=2,
                                      axis=1)
            eps = tf.random.normal(shape=(num_labels,))
            repar_logits = eps * tf.exp(logvar * .5) + logits
            variance = tf.exp(logvar)
            if is_training:
                decode_layer_1 = tf.keras.layers.Dense(
                    num_labels,
                    activation="relu",
                    kernel_initializer=modeling.create_initializer())(
                        repar_logits)
                decode_layer_2 = tf.keras.layers.Dense(
                    hidden_size,
                    activation="relu",
                    kernel_initializer=modeling.create_initializer())(
                        decode_layer_1)
                embb_logits = tf.keras.layers.Dense(
                    hidden_size,
                    kernel_initializer=modeling.create_initializer())(
                        decode_layer_2)
                mse = tf.losses.mean_squared_error(
                    labels=output_layer,
                    predictions=embb_logits,
                    reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
                mse = tf.reduce_mean(mse)

        probabilities = tf.nn.sigmoid(logits)
        if finetune_module not in ["simple_repar", "vae"]:
            variance = 0 * logits
        tf.logging.info(
            f"num_labels:{num_labels};logits:{logits};labels:{labels}")
        loss = tf.compat.v1.losses.sigmoid_cross_entropy(
            labels,
            logits,
            weights=label_mask,
            reduction=tf.compat.v1.losses.Reduction.SUM_OVER_BATCH_SIZE)
        if finetune_module == "bnn":
            loss += kl
        if finetune_module == "vae" and is_training:
            loss += mse

        return (loss, logits, probabilities, variance)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings, freeze_bert, finetune_module,
                     num_train_examples):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        label_mask = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"],
                                      dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids)[0], dtype=tf.float32)
        if "label_mask" in features:
            label_mask = tf.cast(features["label_mask"], dtype=tf.float32)
        else:
            label_mask = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, logits, probabilities, variance) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            label_ids, num_labels, use_one_hot_embeddings, freeze_bert,
            finetune_module, num_train_examples, is_real_example, label_mask)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(total_loss, learning_rate,
                                                     num_train_steps,
                                                     num_warmup_steps, use_tpu)

            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                train_op=train_op,
                scaffold_fn=scaffold_fn)

        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(label_ids, logits, probabilities, variance,
                          is_real_example):

                def hemming_loss(labels,
                                 probabilities,
                                 weights=None,
                                 metrics_collections=None,
                                 updates_collections=None,
                                 name=None):
                    probabilities.get_shape().assert_is_compatible_with(
                        labels.get_shape())
                    prob = tf.cast(probabilities, dtype=tf.float32)
                    lab = tf.cast(labels, dtype=tf.float32)
                    total_error = tf.reduce_sum(
                        tf.abs(lab - prob) * is_real_example)
                    h_loss, update_op = tf.metrics.mean(total_error)

                    if metrics_collections:
                        tf.compat.v1.add_to_collections(metrics_collections,
                                                        h_loss)

                    if updates_collections:
                        tf.compat.v1.add_to_collections(updates_collections,
                                                        update_op)

                    return h_loss, update_op

                predictions = tf.cast(tf.round(probabilities), dtype=tf.int32)
                label_ids = tf.cast(label_ids, dtype=tf.int32)
                pred_split = tf.split(predictions, num_labels, axis=-1)
                probs_split = tf.split(probabilities, num_labels, axis=-1)
                label_ids_split = tf.split(label_ids, num_labels, axis=-1)
                variance_split = tf.split(variance, num_labels, axis=-1)

                eval_dict = dict()
                for j in range(num_labels):
                    eval_dict[LABEL_COLUMN[j] + ' variance'] = tf.metrics.mean(
                        variance_split[j], weights=is_real_example)
                    eval_dict[LABEL_COLUMN[j] +
                              ' accuracy'] = tf.metrics.accuracy(
                                  label_ids_split[j],
                                  pred_split[j],
                                  weights=is_real_example)
                    eval_dict[LABEL_COLUMN[j] + ' auc'] = tf.metrics.auc(
                        label_ids_split[j],
                        probs_split[j],
                        weights=is_real_example)
                    eval_dict[LABEL_COLUMN[j] +
                              ' f1'] = tf.contrib.metrics.f1_score(
                                  label_ids_split[j],
                                  probs_split[j],
                                  weights=is_real_example)
                    eval_dict[LABEL_COLUMN[j] + ' recall'] = tf.metrics.recall(
                        label_ids_split[j],
                        pred_split[j],
                        weights=is_real_example)
                    eval_dict[LABEL_COLUMN[j] +
                              ' precision'] = tf.metrics.precision(
                                  label_ids_split[j],
                                  pred_split[j],
                                  weights=is_real_example)
                    eval_dict[
                        LABEL_COLUMN[j] +
                        ' recall_at_precision_90'] = tf.contrib.metrics.recall_at_precision(
                            label_ids_split[j],
                            probs_split[j],
                            0.9,
                            weights=is_real_example)
                    eval_dict[
                        LABEL_COLUMN[j] +
                        ' recall_at_precision_95'] = tf.contrib.metrics.recall_at_precision(
                            label_ids_split[j],
                            probs_split[j],
                            0.95,
                            weights=is_real_example)
                    eval_dict[LABEL_COLUMN[j] +
                              ' true_positives'] = tf.metrics.true_positives(
                                  label_ids_split[j],
                                  pred_split[j],
                                  weights=is_real_example)
                    eval_dict[LABEL_COLUMN[j] +
                              ' false_positives'] = tf.metrics.false_positives(
                                  label_ids_split[j],
                                  pred_split[j],
                                  weights=is_real_example)
                    eval_dict[LABEL_COLUMN[j] +
                              ' true_negatives'] = tf.metrics.true_negatives(
                                  label_ids_split[j],
                                  pred_split[j],
                                  weights=is_real_example)
                    eval_dict[LABEL_COLUMN[j] +
                              ' false_negatives'] = tf.metrics.false_negatives(
                                  label_ids_split[j],
                                  pred_split[j],
                                  weights=is_real_example)

                eval_dict['hemming_loss'] = hemming_loss(
                    label_ids, probabilities, weights=is_real_example)
                eval_dict["mean_variance"] = tf.metrics.mean(
                    values=variance, weights=is_real_example)
                return eval_dict

            eval_metrics = (metric_fn, [
                label_ids, logits, probabilities, variance, is_real_example
            ])
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                loss=total_loss,
                eval_metrics=eval_metrics,
                scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
                mode=mode,
                predictions={
                    "probs": probabilities,
                    "logits": logits,
                    "variance": variance
                },
                scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn


def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(all_input_ids,
                            shape=[num_examples, seq_length],
                            dtype=tf.int32),
            "input_mask":
                tf.constant(all_input_mask,
                            shape=[num_examples, seq_length],
                            dtype=tf.int32),
            "segment_ids":
                tf.constant(all_segment_ids,
                            shape=[num_examples, seq_length],
                            dtype=tf.int32),
            "label_ids":
                tf.constant(all_label_ids,
                            shape=[num_examples,
                                   len(LABEL_COLUMN)],
                            dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn


def prepare_tfrecords(task_name,
                      data_dir,
                      max_seq_length,
                      tokenizer,
                      use_tpu,
                      batch_size,
                      sample_size=1,
                      convert_tsv=False):
    tf.logging.info("Reading %s/%s.tsv", data_dir, task_name)
    df = pd.read_csv(f"{data_dir}/{task_name}.tsv", sep="\t")
    record_path = f"{data_dir}/{task_name}-{sample_size}.tfrecord"
    num_actual_examples = len(df)
    if convert_tsv or (not tf.io.gfile.exists(record_path)):
        examples = [
            InputExample(guid=x[ID_COLUMN],
                         text_a=x[DATA_COLUMN],
                         text_b=None,
                         label=x[LABEL_COLUMN].tolist(),
                         label_mask=ast.literal_eval(x[MASK_COLUMN])
                         if MASK_COLUMN in x else None)
            for _, x in df.iterrows()
        ]
        examples *= sample_size
        if use_tpu:
            while len(examples) % batch_size != 0:
                examples.append(PaddingInputExample())
            assert len(examples) % batch_size == 0
        num_steps_per_epoch = int(len(examples) // batch_size)
        num_padded_examples = len(examples) - num_actual_examples * sample_size
        file_based_convert_examples_to_features(examples, LABEL_COLUMN,
                                                max_seq_length, tokenizer,
                                                record_path)
    else:
        num_padded_examples = 0
        if use_tpu:
            remainder = (num_actual_examples * sample_size) % batch_size
            if remainder > 0:
                num_padded_examples = batch_size - remainder
        num_steps_per_epoch = int(
            (num_actual_examples + num_padded_examples) // batch_size)

    return df, num_actual_examples, num_padded_examples, num_steps_per_epoch


def eval_routine(task_name,
                 data_dir,
                 output_dir,
                 max_seq_length,
                 tokenizer,
                 estimator,
                 use_tpu,
                 batch_size,
                 ckpt_steps,
                 sample_size=1,
                 convert_tsv=False):
    eval_task_name = task_name if task_name else "eval"
    _, num_examples, num_padded_examples, num_steps = prepare_tfrecords(
        eval_task_name, data_dir, max_seq_length, tokenizer, use_tpu,
        batch_size, 1, convert_tsv)
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    num_padded_examples + num_examples, num_examples,
                    num_padded_examples)
    tf.logging.info("  Batch size = %d", batch_size)
    if use_tpu:
        tf.logging.info("  Num steps = %d", num_steps)
    writer = tf.io.gfile.GFile(f"{output_dir}/{eval_task_name}-results.txt",
                               "w+")
    eval_input_fn = file_based_input_fn_builder(
        input_file=f"{data_dir}/{eval_task_name}-1.tfrecord",
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=use_tpu)

    if tf.io.gfile.exists(f"{output_dir}/model.ckpt-best.meta"):
        return estimator.evaluate(
            input_fn=eval_input_fn,
            steps=num_steps,
            checkpoint_path=f"{output_dir}/model.ckpt-best")

    key_name = "loss"
    best_result = {key_name: np.inf}
    for s in ckpt_steps:
        tf.logging.info(f"***** Eval results for step {s} *****")
        writer.write(f"***** Eval results for step {s} *****\n")
        checkpoint_path = f"{output_dir}/model.ckpt-{s}"
        result = estimator.evaluate(input_fn=eval_input_fn,
                                    steps=num_steps,
                                    checkpoint_path=checkpoint_path)
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            if result[key_name] < best_result[key_name]:
                best_result = result
                for ext in ["meta", "data-00000-of-00001", "index"]:
                    src_ckpt = checkpoint_path + ".{}".format(ext)
                    tgt_ckpt = checkpoint_path.rsplit(
                        "-", 1)[0] + "-best.{}".format(ext)
                    tf.logging.info("saving {} to {}".format(
                        src_ckpt, tgt_ckpt))
                    tf.io.gfile.copy(src_ckpt, tgt_ckpt, overwrite=True)
                    writer.write("saved {} to {}\n".format(src_ckpt, tgt_ckpt))

    return best_result


def predict_routine(task_name,
                    data_dir,
                    output_dir,
                    max_seq_length,
                    tokenizer,
                    estimator,
                    use_tpu,
                    batch_size,
                    sample_size=1,
                    convert_tsv=False):
    predict_task_name = task_name if task_name else "predict"
    predict_df, num_actual_examples, num_padded_examples, _ = prepare_tfrecords(
        predict_task_name, data_dir, max_seq_length, tokenizer, use_tpu,
        batch_size, sample_size, convert_tsv)
    tf.logging.info("***** Running prediction *****")
    tf.logging.info("  Num examples = %d (%d x %d actual, %d padding)",
                    num_padded_examples + num_actual_examples * sample_size,
                    num_actual_examples, sample_size, num_padded_examples)
    tf.logging.info("  Batch size = %d", batch_size)
    tf.logging.info("  Input = %s",
                    f"{data_dir}/{predict_task_name}-{sample_size}.tfrecord")
    predict_input_fn = file_based_input_fn_builder(
        input_file=f"{data_dir}/{predict_task_name}-{sample_size}.tfrecord",
        seq_length=max_seq_length,
        is_training=False,
        drop_remainder=use_tpu)
    checkpoint_path = f"{output_dir}/model.ckpt-best"
    output = estimator.predict(input_fn=predict_input_fn,
                               checkpoint_path=checkpoint_path)
    probs = []
    logits = []
    per_iter_probs = []
    per_iter_logits = []
    for (i, pred) in enumerate(output):
        if i > 0 and i % num_actual_examples == 0:
            probs.append(per_iter_probs)
            logits.append(per_iter_logits)
            per_iter_probs = []
            per_iter_logits = []
        if i >= num_actual_examples * sample_size:
            break
        per_iter_probs.append(pred["probs"])
        per_iter_logits.append(pred["logits"])
    probs = np.array(probs, dtype=np.float)
    logits = np.array(logits, dtype=np.float)
    mean_probs = probs.mean(axis=0)
    mean_logits = logits.mean(axis=0)
    variance = logits.var(axis=0)
    output = [{
        ID_COLUMN: predict_df[ID_COLUMN][i],
        LANG_COLUMN: predict_df[LANG_COLUMN][i],
        "probs": mean_probs[i],
        "variance": variance[i],
        "logits": mean_logits[i],
        "raw_probs": probs[:, i, :]
    } for i in range(len(mean_probs))]
    return output
