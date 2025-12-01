"""
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
"""

from ast import Dict
import numpy as np
import random as rd
import scipy.sparse as sp
from time import time
import os

import torch


class DataHandler(object):
    def __init__(self, dataset, batch_size):
        self.dataset_name = dataset
        self.batch_size = batch_size
        if self.dataset_name.find("Taobao") != -1:
            behs = ["pv", "cart", "train"]

        elif self.dataset_name.find("Tmall") != -1:
            behs = ["pv", "fav", "cart", "train"]

        elif self.dataset_name.find("Jdata") != -1:
            behs = ["view", "collect", "cart", "train"]

        else:
            raise ValueError("Invalid dataset name.")
        self.predir = "dataset/" + self.dataset_name
        self.behs = behs
        self.beh_num = len(behs)

        self.trnMats = None
        self.tstMats = None
        self.trnDicts = None
        self.trnDicts_item = None
        self.tstDicts = None
        self.allInter = dict()
        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = []

        self.LoadData()

    def LoadData(self):
        for i in range(len(self.behs) - 1):
            beh = self.behs[i]
            file_name = self.predir + "/" + beh + ".txt"
            with open(file_name) as f:
                for l in f.readlines():
                    if len(l) > 0:
                        l = l.strip("\n").split(" ")
                        items = [int(i) for i in l[1:]]
                        uid = int(l[0])
                        self.n_items = max(self.n_items, max(items))
                        self.n_users = max(self.n_users, uid)

        train_file = self.predir + "/train.txt"
        test_file = self.predir + "/test.txt"
        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n").split(" ")
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    self.exist_users.append(uid)
                    self.n_items = max(self.n_items, max(items))
                    self.n_users = max(self.n_users, uid)
                    self.n_train += len(items)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n")
                    try:
                        items = [int(i) for i in l.split(" ")[1:]]
                    except Exception:
                        continue
                    self.n_items = max(self.n_items, max(items))
                    self.n_test += len(items)
        self.n_items += 1
        self.n_users += 1

        self.trnMats = [
            sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
            for i in range(len(self.behs))
        ]
        self.trainAll = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.tstMats = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.trnDicts = [dict() for i in range(len(self.behs))]
        self.trnDicts_item = [dict() for i in range(len(self.behs))]
        self.tstDicts = dict()
        self.interNum = [0 for i in range(len(self.behs))]
        self.edge_index = []
        self.edge_type = []
        for i in range(len(self.behs)):
            row = []
            col = []

            beh = self.behs[i]
            beh_filename = self.predir + "/" + beh + ".txt"
            with open(beh_filename) as f:
                for l in f.readlines():
                    if len(l) == 0:
                        break
                    l = l.strip("\n")
                    items = [int(i) for i in l.split(" ")]
                    uid, items_list = items[0], items[1:]
                    self.interNum[i] += len(items_list)
                    for item in items_list:
                        self.trnMats[i][uid, item] = 1.0
                        self.trainAll[uid, item] = 1.0
                        row.append(uid)
                        col.append(self.n_users + item)
                    self.trnDicts[i][uid] = items_list
                row += col
                col += row[: len(col)]
                edge = torch.tensor([row, col], dtype=torch.long)
                self.edge_index.append(edge)
                self.edge_type.append(torch.tensor([i for _ in range(len(row))]))

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) == 0:
                    break
                l = l.strip("\n")
                try:
                    items = [int(i) for i in l.split(" ")]
                except Exception:
                    continue
                uid, test_items = items[0], items[1:]

                for i in test_items:
                    self.tstMats[uid, i] = 1.0
                self.tstDicts[uid] = test_items
        for dic in self.trnDicts:
            for u, items in dic.items():
                if u in self.allInter.keys():
                    self.allInter[u].extend(items)
                else:
                    self.allInter[u] = list(items)
        self.print_statistics()
        self.train_items = self.trnDicts[-1]
        self.test_set = self.tstDicts
        self.test_gt_length = np.array([len(x) for _,x in self.test_set.items()])
        self.path = self.predir

    def sample_behvaior_item(self, u, beh, num):
        pos_items = []
        neg_items = []
        if u in self.trnDicts[beh].keys():

            while len(pos_items) < num:
                items = rd.choice(self.trnDicts[beh][u])
                pos_items.append(items)
            while len(neg_items) < num:
                neg = rd.randint(0, self.n_items-1)
                if neg not in self.allInter[u] and neg not in neg_items:
                    neg_items.append(neg)
        else:
            pos_items = [self.n_items] * num
            neg_items = [self.n_items] * num
        return pos_items, neg_items

    def sample_pos_items(self, u, num):
        pos_items = self.train_items[u]
        n_pos_items = len(pos_items)
        pos_batch = []
        while True:
            if len(pos_batch) == num:
                break
            pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_i_id = pos_items[pos_id]

            if pos_i_id not in pos_batch:
                pos_batch.append(pos_i_id)
        return pos_batch

    def sample_neg_items(self, u, num):
        neg_items = []
        while True:
            if len(neg_items) == num:
                break
            neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_id not in self.allInter[u] and neg_id not in neg_items:
                neg_items.append(neg_id)
        return neg_items

    def sample(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.exist_users, self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def sample_test(self):
        if self.batch_size <= self.n_users:
            users = rd.sample(self.test_set.keys(), self.batch_size)
        else:
            users = [rd.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.test_set[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num:
                    break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num:
                    break
                neg_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
                if (
                    neg_id not in (self.test_set[u] + self.train_items[u])
                    and neg_id not in neg_items
                ):
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(u, num):
            neg_items = list(set(self.neg_pools[u]) - set(self.train_items[u]))
            return rd.sample(neg_items, num)

        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items

    def print_statistics(self):
        num_users, num_items = self.n_users, self.n_items
        num_ratings = sum(self.interNum)
        density = 1.0 * num_ratings / (num_users * num_items)
        sparsity = 1 - density
        data_info = [
            "Dataset name: %s" % self.dataset_name,
            "The number of users: %d" % num_users,
            "The number of items: %d" % num_items,
            "The behavior ratings: {}".format(self.interNum),
            "The number of ratings: %d" % num_ratings,
            "Average actions of users: %.2f" % (1.0 * num_ratings / num_users),
            "Average actions of items: %.2f" % (1.0 * num_ratings / num_items),
            "The density of the dataset: %.6f" % (density),
            "The sparsity of the dataset: %.6f%%" % (sparsity * 100),
        ]
        data_info = "\n".join(data_info)
        print(data_info)

        # print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        # print('n_interactions=%d' % (self.n_train + self.n_test))
        # print('n_train=%d, n_test=%d, sparsity=%.5f' % (
        # self.n_train, self.n_test, (self.n_train + self.n_test) / (self.n_users * self.n_items)))

    def get_sparsity_split(self):
        try:
            split_uids, split_state = [], []
            lines = open(self.path + "/sparsity.split", "r").readlines()

            for idx, line in enumerate(lines):
                if idx % 2 == 0:
                    split_state.append(line.strip())
                    print(line.strip())
                else:
                    split_uids.append([int(uid) for uid in line.strip().split(" ")])
            print("get sparsity split.")

        except Exception:
            split_uids, split_state = self.create_sparsity_split()
            f = open(self.path + "/sparsity.split", "w")
            for idx in range(len(split_state)):
                f.write(split_state[idx] + "\n")
                f.write(" ".join([str(uid) for uid in split_uids[idx]]) + "\n")
            print("create sparsity split.")

        return split_uids, split_state

    def create_sparsity_split(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)

        split_uids = list()  # [[temp0], [temp1], ..]  temp0: user list[u1, u2..]

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = self.n_train
        n_rates = 0

        split_state = []
        temp0 = []
        temp1 = []
        temp2 = []
        temp3 = []
        temp4 = []

        # print user_n_iid

        for idx, n_iids in enumerate(sorted(user_n_iid)):
            if n_iids < 9:
                temp0 += user_n_iid[n_iids]
            elif n_iids < 13:
                temp1 += user_n_iid[n_iids]
            elif n_iids < 17:
                temp2 += user_n_iid[n_iids]
            elif n_iids < 20:
                temp3 += user_n_iid[n_iids]
            else:
                temp4 += user_n_iid[n_iids]

        split_uids.append(temp0)
        split_uids.append(temp1)
        split_uids.append(temp2)
        split_uids.append(temp3)
        split_uids.append(temp4)
        split_state.append("#users=[%d]" % (len(temp0)))
        split_state.append("#users=[%d]" % (len(temp1)))
        split_state.append("#users=[%d]" % (len(temp2)))
        split_state.append("#users=[%d]" % (len(temp3)))
        split_state.append("#users=[%d]" % (len(temp4)))

        return split_uids, split_state

    def create_sparsity_split2(self):
        all_users_to_test = list(self.test_set.keys())
        user_n_iid = dict()

        # generate a dictionary to store (key=n_iids, value=a list of uid).
        for uid in all_users_to_test:
            train_iids = self.train_items[uid]
            test_iids = self.test_set[uid]

            n_iids = len(train_iids) + len(test_iids)

            if n_iids not in user_n_iid.keys():
                user_n_iid[n_iids] = [uid]
            else:
                user_n_iid[n_iids].append(uid)
        split_uids = list()

        # split the whole user set into four subset.
        temp = []
        count = 1
        fold = 4
        n_count = self.n_train + self.n_test
        n_rates = 0

        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.25 * (self.n_train + self.n_test):
                split_uids.append(temp)

                state = "#inter per user<=[%d], #users=[%d], #all rates=[%d]" % (
                    n_iids,
                    len(temp),
                    n_rates,
                )
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)

                state = "#inter per user<=[%d], #users=[%d], #all rates=[%d]" % (
                    n_iids,
                    len(temp),
                    n_rates,
                )
                split_state.append(state)
                print(state)

        return split_uids, split_state
