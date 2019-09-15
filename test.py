"""Could test best model provided by us, in for loop."""
import pickle
import torch
import os
from Evaluate import Evaluate
from config import config
from model import TOICNN

mode = "test"  # test dev
test_best = True
epoch_start = 1
epoch_end = 50

config.if_detail = True
config.if_output = False
config.if_filter = True
config.score_th = 0.5
misc_config = pickle.load(open(config.get_pkl_path("config"), "rb"))
config.load_config(misc_config)

with open(config.get_pkl_path(mode), "rb") as f:
    word_batches, char_batches, char_len_batches, pos_tag_batches, entity_batches, toi_batches = pickle.load(f)
print("load data from " + config.get_pkl_path(mode))

for e in range(epoch_start, epoch_end + 1):
    model_path = config.get_model_path() + f"epoch{e}.pth"  # test trained model
    # model_path = config.get_model_path() + f"best.pth"  # test best model
    if not os.path.exists(model_path):
        continue
    print("load model from " + model_path)
    ner_model = TOICNN(config)
    ner_model.load_state_dict(torch.load(model_path))
    if config.if_gpu and torch.cuda.is_available():
        ner_model = ner_model.cuda()
    evaluate = Evaluate(ner_model, config)

    evaluate.get_f1(zip(word_batches, char_batches, char_len_batches, pos_tag_batches, entity_batches, toi_batches))
    print("\n\n")
