import pickle
import time
import numpy as np
import torch

from torch.autograd import Variable
from random import shuffle
from Evaluate import Evaluate
from config import config
from model import TOICNN
from until import create_opt, select_toi, adjust_learning_rate, generate_mask


with open(config.get_pkl_path("train"), "rb") as f:
    train_word_batches, train_char_batches, train_char_len_batches, train_pos_tag_batches, train_entity_batches, train_toi_batches = pickle.load(f)
with open(config.get_pkl_path("dev"), "rb") as f:
    dev_word_batches, dev_char_batches, dev_char_len_batches, dev_pos_tag_batches, dev_entity_batches, dev_toi_batches = pickle.load(f)


misc_config = pickle.load(open(config.get_pkl_path("config"), "rb"))
config.load_config(misc_config)

ner_model = TOICNN(config)
ner_model.load_vector()
if config.if_gpu and torch.cuda.is_available():
    ner_model = ner_model.cuda()

evaluate = Evaluate(ner_model, config)

parameters = filter(lambda p: p.requires_grad, ner_model.parameters())
optimizer = create_opt(parameters, config.opt, config.lr)

best_model = None
best_per = 0
pre_loss = 100000
train_all_batches = list(zip(train_word_batches, train_char_batches, train_char_len_batches, train_pos_tag_batches, train_entity_batches, train_toi_batches))

for e_ in range(config.epoch):
    print("Epoch:", e_ + 1)

    cur_time = time.time()
    if config.if_shuffle:
        shuffle(train_all_batches)

    losses = []
    for word_batch, char_batch, char_len_batch, pos_tag_batch, entity_batch, toi_batch in train_all_batches:

        word_batch_var = Variable(torch.LongTensor(np.array(word_batch)))
        mask_batch_var = generate_mask(word_batch_var.shape)
        char_batch_var = Variable(torch.LongTensor(np.array(char_batch)))
        pos_tag_batch_var = Variable(torch.LongTensor(np.array(pos_tag_batch)))
        toi_box_batch, label_batch = select_toi(toi_batch)
        gold_label_vec = Variable(torch.LongTensor(np.hstack(label_batch)))
        if config.if_gpu:
            word_batch_var = word_batch_var.cuda()
            mask_batch_var = mask_batch_var.cuda()
            char_batch_var = char_batch_var.cuda()
            pos_tag_batch_var = pos_tag_batch_var.cuda()
            gold_label_vec = gold_label_vec.cuda()

        ner_model.train()
        optimizer.zero_grad()
        cls_s, _ = ner_model(mask_batch_var, word_batch_var, char_batch_var, char_len_batch, pos_tag_batch_var, toi_box_batch)
        loss = ner_model.calc_loss(cls_s, gold_label_vec)

        loss.backward()
        # torch.nn.utils.clip_grad_norm_(ner_model.parameters(), 3, norm_type=2)
        optimizer.step()

        losses.append(loss.data.cpu().numpy())

    sub_loss = np.mean(losses)
    print(f'Avg loss = {sub_loss:.4f}')
    print(f"Training step took {time.time() - cur_time:.0f} seconds")
    if e_ >= 20:
        print("Dev:")
        cls_f1 = evaluate.get_f1(zip(dev_word_batches, dev_char_batches, dev_char_len_batches, dev_pos_tag_batches, dev_entity_batches, dev_toi_batches))
        if cls_f1 > best_per:
            best_per = cls_f1
            model_path = config.get_model_path() + f"epoch{e_ + 1}.pth"
            torch.save(ner_model.state_dict(), model_path)
            print("model save in " + model_path)
        print('\n\n')

    if sub_loss >= pre_loss:
        adjust_learning_rate(optimizer)
    pre_loss = sub_loss
