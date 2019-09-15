import os

vec_file = {"wiki": "./model/word2vec/wikipedia-pubmed-and-PMC-w2v.bin",
            "glove": "./model/word2vec/glove_word2vec.txt",
            "PubMed": "./model/word2vec/PubMed-shuffle-win-2.bin"}


class Config:
    def __init__(self):
        # sample
        self.data_set = "ACE05"  # ACE05 ACE04 GENIA
        self.data_path = f"./dataset/{self.data_set}/"
        self.C = 10  # 10 8 6
        self.train_pos_iou_th = 1
        self.train_neg_iou_th = 0.67  # 0.81 0.67 0.76 0.51 0.01

        # model
        self.vec_model = "glove"  # glove wiki PubMed
        self.word_embedding_size = 100 if self.vec_model == "glove" else 200
        self.word2vec_path = vec_file[self.vec_model]
        self.if_char = True
        self.char_embedding_size = 25
        self.if_pos = True
        self.pos_embedding_size = 6
        self.if_transformer = True
        self.N = 2
        self.h = 4
        self.if_bidirectional = True
        self.kernel_size = 5
        self.feature_maps_size = 36

        # train
        self.if_gpu = True
        self.if_shuffle = True
        self.if_freeze = False
        self.dropout = 0.5
        self.epoch = 100
        self.batch_size = 8
        self.opt = "Adam"
        self.lr = 3e-4  # 0.005 1e-4

        # test
        self.if_output = False
        self.if_detail = False
        self.if_filter = True
        self.score_th = 0.5

    def __repr__(self):
        return str(vars(self))

    def get_pkl_path(self, mode):
        path = self.data_path
        if mode == "word2vec":
            path += f"word_vec"
        else:
            if mode == "config":
                path += f"config"
            else:
                path += mode + "/" + mode
        return path + f"_{self.vec_model}.pkl"

    def get_model_path(self):
        path = f"./model/{self.data_set}/"
        path += f"{self.vec_model}"
        if not os.path.exists(path):
            os.makedirs(path)
        return path + "/"

    def get_result_path(self):
        path = f"./result/{self.data_set}"
        if not os.path.exists(path):
            os.makedirs(path)
        path += f"/{self.vec_model}"
        return path + ".data"

    def load_config(self, misc_dict):
        self.word_kinds = misc_dict["word_kinds"]
        self.char_kinds = misc_dict["char_kinds"]
        self.pos_tag_kinds = misc_dict["pos_tag_kinds"]
        self.label_kinds = misc_dict["label_kinds"]
        self.id2label = misc_dict["id2label"]

        print(self)
        self.id2word = misc_dict["id2word"]


config = Config()
