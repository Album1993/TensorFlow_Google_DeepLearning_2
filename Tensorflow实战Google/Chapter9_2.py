import codecs
import collections

from operator import itemgetter

RAW_DATA_TRAIN = "./Deep_Learning_with_TensorFlow/datasets/PTB_data/ptb.train.txt"

VOCAB_OUTPUT_TRAIN = "./Result/chapter9_2_ptb_train.vocab"

RAW_DATA_TEST = "./Deep_Learning_with_TensorFlow/datasets/PTB_data/ptb.test.txt"

VOCAB_OUTPUT_TEST = "./Result/chapter9_2_ptb_test.vocab"

RAW_DATA_VALID = "./Deep_Learning_with_TensorFlow/datasets/PTB_data/ptb.valid.txt"

VOCAB_OUTPUT_VALID = "./Result/chapter9_2_ptb_valid.vocab"

# counter = collections.Counter()
#
# with codecs.open(RAW_DATA,"r","utf-8") as f:
#     for line in f:
#         for word in line.strip().split():
#             counter[word] += 1
#
# sorted_word_to_cnt = sorted(counter.items(),key=itemgetter(1),reverse=True)
#
# sorted_words = [x[0] for x in sorted_word_to_cnt]
#
# with codecs.open(VOCAB_OUTPUT,'w','utf-8') as file_output:
#     for word in sorted_words:
#         file_output.write(word + '\n')


def generate(raw_data,output):
    counter = collections.Counter()

    with codecs.open(raw_data, "r", "utf-8") as f:
        for line in f:
            for word in line.strip().split():
                counter[word] += 1

    sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)

    sorted_words = [x[0] for x in sorted_word_to_cnt]

    with codecs.open(output, 'w', 'utf-8') as file_output:
        for word in sorted_words:
            file_output.write(word + '\n')



generate(RAW_DATA_TRAIN,VOCAB_OUTPUT_TRAIN)
generate(RAW_DATA_TEST,VOCAB_OUTPUT_TEST)
generate(RAW_DATA_VALID,VOCAB_OUTPUT_VALID)