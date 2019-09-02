# import gzip
import gensim
import logging
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET

logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO)


def create_corpus(tag_list, doc):
    list_val = []
    for tag_name in tag_list:
        tag_property = doc.getElementsByTagName(tag_name)
        for skill in tag_property:
            try:
                list_val.append(tag_name + ' ' + skill.firstChild.nodeValue.rstrip())
            except AttributeError:
                continue
    return list_val


def create_input_file(file_name):
    f = open(file_name, "r")
    fl = f.readlines()
    fw = open("input_data.txt", "w")
    for line in fl:
        line = line[:-1]
        print(line)
        xml_tree = ET.parse(line)  # ### Define line
        elem_list = []
        for elem in xml_tree.iter():
            elem_list.append(elem.tag)

        elem_list = list(set(elem_list))
        doc = xml.dom.minidom.parse(line)
        corpus_list = create_corpus(elem_list, doc)

        for corpus in corpus_list:
            fw.write(corpus)
            fw.write("\n")
    f.close()
    fw.close()
    return


def show_file_contents(input_file):
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):
            print(line)
            break


def read_input(input_file):
    """This method reads the input file"""

    logging.info("reading file {0}...this may take a while".format(input_file))
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if i % 10000 == 0:
                logging.info("read {0} reviews".format(i))
            # do some pre-processing and return list of words for each review
            # text
            yield gensim.utils.simple_preprocess(line)


if __name__ == '__main__':

    create_input_file("Datasets/courses_schemas.txt")
    # create_input_file("Datasets/real_es_schema.txt")

    abspath = os.path.dirname(os.path.abspath(__file__))
    # data_file = os.path.join(abspath, "reviews_data.txt")
    data_file = os.path.join(abspath, "input_data.txt")

    # read the tokenized reviews into a list
    # each review item becomes a series of words
    # so this becomes a list of lists
    documents = list(read_input(data_file))
    logging.info("Done reading data file")

    # build vocabulary and train model
    model = gensim.models.Word2Vec(
        documents,
        size=50,
        window=10,
        min_count=2,
        workers=10)
    model.train(documents, total_examples=len(documents), epochs=10)

    # save only the word vectors
    # model.wv.save(os.path.join(abspath, "/vectors/default"))
    model.wv.save("/Users/ayodhya/Documents/GitHub/Data_mapping/word2vec_vectors")

    # wv = KeyedVectors.load("/Users/ayodhya/PycharmProjects/word2vec/vectors/default", mmap='r')
    #
    # w1 = "dirty"
    # print("Most similar to {0}".format(w1), wv.most_similar(positive=w1))
    #
    # # look up top 6 words similar to 'polite'
    # w1 = ["polite"]
    # print(
    #     "Most similar to {0}".format(w1),
    #     wv.most_similar(
    #         positive=w1,
    #         topn=6))

    # # look up top 6 words similar to 'france'
    # w1 = ["france"]
    # print(
    #     "Most similar to {0}".format(w1),
    #     model.wv.most_similar(
    #         positive=w1,
    #         topn=6))
    #
    # # look up top 6 words similar to 'shocked'
    # w1 = ["shocked"]
    # print(
    #     "Most similar to {0}".format(w1),
    #     model.wv.most_similar(
    #         positive=w1,
    #         topn=6))
    #
    # # look up top 6 words similar to 'shocked'
    # w1 = ["beautiful"]
    # print(
    #     "Most similar to {0}".format(w1),
    #     model.wv.most_similar(
    #         positive=w1,
    #         topn=6))

    # # get everything related to stuff on the bed
    # w1 = ["bed", 'sheet', 'pillow']
    # w2 = ['couch']
    # print(
    #     "Most similar to {0}".format(w1),
    #     wv.most_similar(
    #         positive=w1,
    #         negative=w2,
    #         topn=10))

    # # similarity between two different words
    # print("Similarity between 'dirty' and 'smelly'",
    #       wv.similarity(w1="dirty", w2="smelly"))

    # # similarity between two identical words
    # print("Similarity between 'dirty' and 'dirty'",
    #       model.wv.similarity(w1="dirty", w2="dirty"))
    #
    # # similarity between two unrelated words
    # print("Similarity between 'dirty' and 'clean'",
    #       model.wv.similarity(w1="dirty", w2="clean"))

    # vector = wv['computer']
