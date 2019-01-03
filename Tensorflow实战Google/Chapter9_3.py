import codecs
import sys

RAW_DATA_TRAIN = "./Deep_Learning_with_TensorFlow/datasets/PTB_data/ptb.train.txt"
VOCAB_TRAIN = "./Result/chapter9_2_ptb_train.vocab"
OUTPUT_DATA_TRAIN = "./Result/chapter9_3_ptb.train"

RAW_DATA_TEST = "./Deep_Learning_with_TensorFlow/datasets/PTB_data/ptb.test.txt"
VOCAB_TEST = "./Result/chapter9_2_ptb_test.vocab"
OUTPUT_DATA_TEST = "./Result/chapter9_3_ptb.test"

RAW_DATA_VALID = "./Deep_Learning_with_TensorFlow/datasets/PTB_data/ptb.valid.txt"
VOCAB_VALID = "./Result/chapter9_2_ptb_valid.vocab"
OUTPUT_DATA_VALID = "./Result/chapter9_3_ptb.valid"

def generate(vocab,rawDate,output ):
    with codecs.open(vocab, "r", "utf-8") as f_vocab:
        vocab = [w.strip() for w in f_vocab.readlines()]

    word_to_id = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

    print(word_to_id)

    def get_id(word):
        return word_to_id[word] if word in word_to_id else word_to_id["<unk>"]

    fin = codecs.open(rawDate, "r", "utf-8")
    fout = codecs.open(output, 'w', 'utf-8')

    for line in fin:
        words = line.strip().split() + ["<eos>"]

        out_line = ' '.join([str(get_id(w)) for w in words]) + '\n'

        fout.write(out_line)

    fin.close()
    fout.close()



generate(VOCAB_TRAIN,RAW_DATA_TRAIN,OUTPUT_DATA_TRAIN)
generate(VOCAB_TEST,RAW_DATA_TEST,OUTPUT_DATA_TEST)
generate(VOCAB_VALID,RAW_DATA_VALID,OUTPUT_DATA_VALID)

