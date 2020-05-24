__author__ = 'JudePark'
__email__ = 'judepark@kookmin.ac.kr'


import pandas as pd


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


def read_nli_examples(input_file: str):
    """
    :param input_files: ***.tsv
    :return: NLIExample
    """

    if input_file.split('.')[-1] != 'tsv':
        raise ValueError('Given data format is tsv.')

    nli_df = pd.read_csv(input_file, sep='\\t', encoding='utf-8')
    nli_examples = [NLIExample(sentence1, sentence2, gold_label)
                    for (sentence1, sentence2, gold_label) in
                    zip(nli_df['sentence_1'], nli_df['sentence_2'], nli_df['gold_label'])]

    return nli_examples


if __name__ == '__main__':
    read_nli_examples('../rsc/data/multinli.train.ko.tsv')