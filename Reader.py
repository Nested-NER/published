import os
import warnings
import pickle
import numpy as np
from collections import namedtuple, defaultdict
from until import calc_iou

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
from gensim.models import KeyedVectors
Sent_info = namedtuple("Sent_info", "words, chars, pos_tags, entities")


class Reader:
    def __init__(self, config):
        self.config = config
        self.UNK = "#UNK#"
        self.infos = {}
        self.max_C = defaultdict(int)
        self.candidation_hit = defaultdict(int)
        self.entity_num = defaultdict(int)
        self.tois_num = defaultdict(int)

    def generate_toi(self, sent_len, entities, mode):
        toi_boxes = [(start, start + length + 1) for start in range(sent_len)
                     for length in range(min(self.config.C, sent_len - start))]
        # if mode == "train":
        #     for boxes in [(e[0], e[1]) for e in entities]:
        #         if boxes not in candidate_boxes:
        #             candidate_boxes.append(boxes)

        return self.toi_labeling(np.array(toi_boxes), entities, mode)

    def toi_labeling(self, toi_boxes, entities, mode):
        ious = calc_iou(toi_boxes, np.array([(e[0], e[1]) for e in entities]))
        max_ious = ious.max(axis=1)
        max_idx = ious.argmax(axis=1)

        tois = []
        for i in range(toi_boxes.shape[0]):
            if max_ious[i] == 1:
                self.candidation_hit[mode] += 1
                tois.append((toi_boxes[i, 0], toi_boxes[i, 1], entities[max_idx[i]][2]))
            elif max_ious[i] >= self.config.train_pos_iou_th and mode == "train":
                tois.append((toi_boxes[i, 0], toi_boxes[i, 1], entities[max_idx[i]][2]))
            elif max_ious[i] < self.config.train_neg_iou_th or mode != "train":
                tois.append((toi_boxes[i, 0], toi_boxes[i, 1], 0))

        self.tois_num[mode] += (len(tois))
        return sorted(tois)

    def read_file(self, file, mode):
        sent_infos = []
        # with open(file, "r", encoding="utf-8") as f:
        with open(file, "r") as f:
            all_infos = f.read().strip().split("\n\n")
            for i, infos in enumerate(all_infos):
                _infos = infos.strip().split("\n")
                words = _infos[0].split()
                chars = [list(t) for t in words]
                pos_tags = _infos[1].split()
                if len(_infos) == 2:
                    entity_list = []
                else:
                    entities = _infos[2].split("|")
                    entity_list = []
                    for entity in entities:
                        positions, label = entity.split()
                        positions = positions.split(",")
                        new_entity = (int(positions[0]), int(positions[1]), label)
                        self.max_C[mode] = max(self.max_C[mode], new_entity[1] - new_entity[0])
                        if new_entity not in entity_list:
                            entity_list.append(new_entity)
                self.entity_num[mode] += len(entity_list)
                sent_infos.append(Sent_info(words, chars, pos_tags, entity_list))
        return sent_infos

    def create_dic(self):
        word_set, char_set, pos_tag_set, label_set = set(), set(), set(), set()
        max_word_len = 0

        for sent_infos in self.infos.values():
            for sent_info in sent_infos:
                for word in sent_info.chars:
                    max_word_len = max(max_word_len, len(word))
                    for char in word:
                        char_set.add(char)
                for word in sent_info.words:
                    word_set.add(word)
                for pos_tag in sent_info.pos_tags:
                    pos_tag_set.add(pos_tag)
                for entity in sent_info.entities:
                    label_set.add(entity[2])

        self.id2word = sorted(list(word_set))
        self.word2id = {}
        self.load_vectors_model()
        self.id2char = sorted(list(char_set))
        self.char2id = {v: i for i, v in enumerate(self.id2char)}
        self.id2pos_tag = sorted(list(pos_tag_set))
        self.pos_tag2id = {v: i for i, v in enumerate(self.id2pos_tag)}
        self.id2label = ["BG"] + sorted((list(label_set)))
        self.label2id = {v: i for i, v in enumerate(self.id2label)}
        self.max_word_len = max_word_len


    def read_all_data(self):
        # for mode in ["train", "test", "dev", "debug"]:
        for mode in ["train", "test", "dev"]:
            self.infos[mode] = self.read_file(self.config.data_path + f"/{mode}/{mode}.data", mode)
        self.create_dic()

    def to_batch(self, mode):
        word_dic = defaultdict(list)
        char_dic = defaultdict(list)
        pos_tag_dic = defaultdict(list)
        entity_dic = defaultdict(list)
        char_len_dic = defaultdict(list)
        toi_dic = defaultdict(list)

        word_batches = []
        char_batches = []
        char_len_batches = []
        pos_tag_batches = []
        entity_batches = []
        toi_batches = []

        for i, sent_info in enumerate(self.infos[mode]):
            entity_vec = [(e[0], e[1], self.label2id[e[2]]) for e in sent_info.entities]
            word_vec = [self.word2id[w] for w in sent_info.words]
            word_num = len(word_vec)

            char_mat = [[self.char2id[c] for c in w] for w in sent_info.chars]
            char_len_vec = [len(w) for w in char_mat]
            pad_id = self.char2id["."]
            char_mat = [w + [pad_id] * (self.max_word_len - len(w)) for w in char_mat]

            pos_tag_vec = [self.pos_tag2id[p] for p in sent_info.pos_tags]

            tois = self.generate_toi(word_num, entity_vec, mode)

            word_dic[word_num].append(word_vec)
            char_dic[word_num].append(char_mat)
            char_len_dic[word_num].append(char_len_vec)
            pos_tag_dic[word_num].append(pos_tag_vec)
            entity_dic[word_num].append(entity_vec)
            toi_dic[word_num].append(tois)

        for length in word_dic.keys():
            word_batch = [word_dic[length][i: i + self.config.batch_size] for i in
                          range(0, len(word_dic[length]), self.config.batch_size)]
            char_batch = [char_dic[length][i: i + self.config.batch_size] for i in
                          range(0, len(char_dic[length]), self.config.batch_size)]
            char_len_batch = [char_len_dic[length][i: i + self.config.batch_size] for i in
                              range(0, len(char_len_dic[length]), self.config.batch_size)]
            pos_tag_batch = [pos_tag_dic[length][i: i + self.config.batch_size] for i in
                             range(0, len(pos_tag_dic[length]), self.config.batch_size)]
            entity_batch = [entity_dic[length][i: i + self.config.batch_size] for i in
                            range(0, len(entity_dic[length]), self.config.batch_size)]
            toi_batch = [toi_dic[length][i: i + self.config.batch_size] for i in
                         range(0, len(toi_dic[length]), self.config.batch_size)]

            word_batches.extend(word_batch)
            char_batches.extend(char_batch)
            char_len_batches.extend(char_len_batch)
            pos_tag_batches.extend(pos_tag_batch)
            entity_batches.extend(entity_batch)
            toi_batches.extend(toi_batch)

        return (word_batches, char_batches, char_len_batches, pos_tag_batches, entity_batches, toi_batches)

    def load_vectors_model(self):
        try:
            vector_model = KeyedVectors.load_word2vec_format(self.config.word2vec_path, binary=True)
        except:
            vector_model = KeyedVectors.load_word2vec_format(self.config.word2vec_path, binary=False)

        unk = np.random.uniform(-0.01, 0.01, self.config.word_embedding_size).astype("float32")
        word2vec = [unk]
        id2word = self.id2word.copy()
        for word in id2word:
            try:
                self.word2id[word] = len(word2vec)
                word2vec.append(vector_model[word])
            except:
                try:
                    word2vec.append(vector_model[word.lower()])
                except:
                    self.id2word.remove(word)
                    self.word2id[word] = 0
        self.id2word = [self.UNK] + self.id2word

        path = self.config.get_pkl_path("word2vec")
        if not os.path.exists(path):
            with open(path, "wb") as f:
                pickle.dump(np.array(word2vec), f)
            print("load vector model form " + self.config.word2vec_path)
