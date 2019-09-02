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
    logging.info("reading file {0}...this may take a while".format(input_file))
    with open(input_file, 'rb') as f:
        for i, line in enumerate(f):

            if i % 10000 == 0:
                logging.info("read {0} reviews".format(i))
            yield gensim.utils.simple_preprocess(line)


if __name__ == '__main__':

    create_input_file("Datasets/courses_schemas.txt")
    # create_input_file("Datasets/real_es_schema.txt")

    abspath = os.path.dirname(os.path.abspath(__file__))
    # data_file = os.path.join(abspath, "reviews_data.txt")
    data_file = os.path.join(abspath, "input_data.txt")

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
    model.wv.save("/Users/ayodhya/Documents/GitHub/Data_mapping/word2vec_vectors")
