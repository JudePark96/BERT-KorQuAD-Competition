__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'

import pandas as pd
import logging

from tqdm import tqdm

logger = logging.getLogger(__name__)


class NLIExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 seq_a,
                 seq_b,
                 gold_label):
        self.seq_a = seq_a
        self.seq_b = seq_b
        self.gold_label = gold_label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "seq_a: %s" % (self.seq_a)
        s += ", seq_b: %s" % (self.seq_b)
        s += ", gold_label: %s" % (self.gold_label)

        return s


class InputFeatures(object):
    """A single set of features of data"""

    def __init__(self, input_id, segment_id, attn_mask) -> None:
        super(InputFeatures, self).__init__()
        self.input_id = input_id
        self.segment_id = segment_id
        self.attn_mask = attn_mask


def read_nli_examples(input_file: str) -> list:
    """
    :param input_files: ***.tsv
    :return: NLIExample
    """

    if input_file.split('.')[-1] != 'tsv':
        raise ValueError('Given data format is tsv.')

    nli_examples = []
    with open(input_file, 'r') as f:
        for idx, data in enumerate(f):
            if idx == 0:
                continue

            parsed = [i.strip() for i in data.split('\t')]
            nli_examples.append(NLIExample(parsed[0], parsed[1], parsed[2]))

    return nli_examples


def convert_examples_to_features(examples, tokenizer, max_seq_length=512) -> list:
    """
    :param examples:
    :param tokenizer:
    :param max_seq_length: 512 를 기준으로 할 것.
    :return:
    """
    features = []

    for (example_idx, example) in tqdm(enumerate(examples)):
        seq_a_tokens = tokenizer.tokenize(str(example.seq_a))
        seq_b_tokens = tokenizer.tokenize(str(example.seq_b))


        if len(seq_a_tokens) + len(seq_b_tokens) + 3 > max_seq_length:
            # skipping
            continue

        tokens = []
        segment_id = []

        tokens.append('[CLS]')
        tokens += seq_a_tokens
        tokens.append('[SEP]')

        for _ in tokens:
            segment_id.append(0)

        sep_idx = len(tokens)

        tokens += seq_b_tokens
        tokens.append('[SEP]')

        for _ in range(sep_idx, len(tokens)):
            segment_id.append(1)

        input_id = tokenizer.convert_tokens_to_ids(tokens)
        attn_mask = [1] * len(input_id)

        while len(input_id) < max_seq_length:
            input_id.append(0)
            attn_mask.append(0)
            segment_id.append(0)

        assert len(input_id) == max_seq_length
        assert len(attn_mask) == max_seq_length
        assert len(segment_id) == max_seq_length

        if example_idx < 20:
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_id]))
            logger.info("attn_mask: %s" % " ".join([str(x) for x in attn_mask]))
            logger.info("segment_id: %s" % " ".join([str(x) for x in segment_id]))

        features.append(
            InputFeatures(
                input_id=input_id,
                segment_id=segment_id,
                attn_mask=attn_mask
            )
        )

    return features

