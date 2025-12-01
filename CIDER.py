import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
from utility.helper import *
from utility.batch_test import *
import random
from datetime import datetime
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
from torch.amp import autocast, GradScaler
from utility.metrics import metrics_dict
scaler = GradScaler()


def get_lables(temp_set, k=0.9999):
    max_item = 0
    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k) - 1]

    print(max_item)
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set

def get_train_instances1(max_item_list, beh_label_list):
    user_train = []
    beh_item_list = [list() for i in range(n_behs)] 

    for i in beh_label_list[-1].keys():
        user_train.append(i)
        beh_item_list[-1].append(beh_label_list[-1][i])
        for j in range(n_behs - 1):
            if not i in beh_label_list[j].keys():
                beh_item_list[j].append([n_items] * max_item_list[j])
            else:
                beh_item_list[j].append(beh_label_list[j][i])

    user_train = np.array(user_train)
    beh_item_list = [np.array(beh_item) for beh_item in beh_item_list]
    return user_train, beh_item_list

def get_train_pairs(user_train_batch, beh_item_tgt_batch):
    u_batch_list, u_global_list, input_i_list = [], [], []
    for i in range(len(user_train_batch)):
        pos_items = beh_item_tgt_batch[i][
            np.where(beh_item_tgt_batch[i] != n_items)
        ]  # ndarray [x,]
        uid = user_train_batch[i][0]
        u_batch_list += [i] * len(pos_items)
        u_global_list += [uid] * len(pos_items)
        input_i_list += pos_items.tolist()

    return (
        np.array(u_batch_list).reshape([-1]),
        np.array(u_global_list).reshape([-1]),
        np.array(input_i_list).reshape([-1]),
    )

class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expansion_factor, dim),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return torch.transpose(self.fn(self.norm(x)) + x, 1, 2)

class CIDER(nn.Module):
    def __init__(self, args, device, max_item_list, dropout=0):
        super(CIDER, self).__init__()
        self.n_behs = len(max_item_list)
        self.embed_dim = args.embed_size
        self.device = device
        self.layers = [nn.ModuleList([]) for i in range(self.n_behs)]
        for beh in range(self.n_behs):
            beh_len = sum(max_item_list[: beh + 1])
            for _ in range(args.layers):
                self.layers[beh].append(
                    nn.Sequential(
                        PreNormResidual(
                            beh_len,
                            FeedForward(beh_len, args.expansion_factor, dropout),
                        ).to(device),
                        PreNormResidual(
                            self.embed_dim * 2,
                            FeedForward(self.embed_dim * 2, args.expansion_factor, dropout),
                        ).to(device),
                    )
                )
        self.layers = nn.ModuleList(self.layers)
        self.item_embeddings = nn.Parameter(torch.FloatTensor(n_items, self.embed_dim))

        self.TR = nn.Linear(self.embed_dim * 3, self.embed_dim)
        self.beh_embeddings = nn.Parameter(torch.FloatTensor(self.n_behs, self.embed_dim))
        self.agg_weights = nn.Parameter(torch.FloatTensor(self.n_behs - 1))
        self.reset_parameters()
        self.apply(self._init_weights)
        self.store_user_embeddings = torch.empty((n_users, self.embed_dim), device=device)
        self.loss = nn.BCEWithLogitsLoss()
    def forward(self, u_batch, beh_batch, beh_sample):
        token_embedding = torch.zeros([1, self.embed_dim], device=device, requires_grad=False)
        item_embeddings = torch.cat([self.item_embeddings, token_embedding], dim=0)
        HBS_embedding = [
            item_embeddings[beh] for beh in beh_batch
        ]  # [[B,max_item1,dim],[B,max_item2,dim],...
        mask = [torch.any((beh) != 0, dim=-1, keepdim=True) for beh in HBS_embedding]
        u_b_embeddings = []

        for beh in range(self.n_behs - 1):
            cat_beh_embedding = self.beh_embeddings[beh] * mask[beh].float()
            HBS_embedding[beh] = torch.cat([HBS_embedding[beh], cat_beh_embedding], dim=-1)
            tmp_e = torch.cat(HBS_embedding[: beh + 1], dim=1).transpose(1, 2)
            layers = self.layers[beh]
            for layer in layers:
                tmp_e = layer(tmp_e)
            u_b_embeddings.append(torch.mean(tmp_e, dim=2))

        denoise_loss = []
        cpt_loss = []
        rel_weight = eval(args.rel_weight)
        denoise_weight = eval(args.denoise_weight)
        cpt_weight = eval(args.cpt_weight)
        coeff = eval(args.coeff)
        for beh, u_b_embedding in enumerate(u_b_embeddings):

            pr_pre = torch.cat(
                [
                    u_b_embedding,
                    self.beh_embeddings[beh].unsqueeze(0).expand(u_b_embedding.shape[0], -1),
                ],
                dim=1,
            )
            pr_next = torch.cat(
                [
                    u_b_embedding,
                    self.beh_embeddings[beh + 1].unsqueeze(0).expand(u_b_embedding.shape[0], -1),
                ],
                dim=1,
            )
            tr_pre = self.TR(pr_pre)
            tr_next = self.TR(pr_next)
            pos_items_pre = beh_sample[beh][0]
            neg_items_pre = beh_sample[beh][1]
            pos_items_next = beh_sample[beh + 1][0]
            neg_items_next = beh_sample[beh + 1][1]

            labels = torch.tensor(
                [1] * len(pos_items_pre) + [0] * len(neg_items_pre),
                dtype=torch.float,
                device=self.device,
            )
            score_pre = torch.cat(
                (
                    torch.sum(tr_pre * item_embeddings[pos_items_pre], dim=-1),
                    torch.sum(tr_pre * item_embeddings[neg_items_pre], dim=-1),
                ),
                dim=0,
            )
            score_next = torch.cat(
                (
                    torch.sum(tr_next * item_embeddings[pos_items_next], dim=-1),
                    torch.sum(tr_next * item_embeddings[neg_items_next], dim=-1),
                ),
                dim=0,
            )

            denoise_loss.append(self.loss(score_pre, labels) * rel_weight[beh])
            cpt_loss.append(self.loss(score_next, labels) * rel_weight[beh + 1])            
            # denoise_loss.append(self.loss(score_pre, labels) * denoise_weight[beh])
            # cpt_loss.append(self.loss(score_next, labels) * cpt_weight[beh])

        stacked_embeddings = torch.stack(u_b_embeddings, dim=0)
        weights = F.softmax(self.agg_weights, dim=0).view(-1, 1, 1)
        tmp_e = (stacked_embeddings * weights).sum(dim=0)

        tmp_e = torch.cat(
            [tmp_e, self.beh_embeddings[-1].unsqueeze(0).expand(tmp_e.shape[0], -1)], dim=1
        )
        target_pos_items = beh_sample[-1][0]
        target_neg_items = beh_sample[-1][1]
        bath_user_embedddings = self.TR(tmp_e)
        bpr = bpr_loss(
            bath_user_embedddings,
            item_embeddings[target_pos_items],
            item_embeddings[target_neg_items],
        )
        self.store_user_embeddings[u_batch] = bath_user_embedddings.detach().float()
        denoise_loss = sum(denoise_loss) * coeff[0]
        cpt_loss = sum(cpt_loss) * coeff[1]
        bpr = bpr * coeff[-1]

        return denoise_loss, cpt_loss, bpr

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.item_embeddings)
        nn.init.xavier_uniform_(self.beh_embeddings)
        nn.init.xavier_uniform_(self.TR.weight)
        nn.init.ones_(self.agg_weights)
        
    def _init_weights(self, module):        
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def predict(self,users):
        scores = torch.matmul(self.store_user_embeddings[users], self.item_embeddings.T)
        return scores
    
    def test_torch(self,
        test_users,
    ):

        u_batch_size = BATCH_SIZE * 2

        n_test_users = len(test_users)
        n_user_batchs = n_test_users // u_batch_size + 1
        test_set = data_generator.test_set
        gt_length = data_generator.test_gt_length
        Ks = eval(args.Ks)
        topk_list = []
        for u_batch_id in range(n_user_batchs):
            start = u_batch_id * u_batch_size
            end = (u_batch_id + 1) * u_batch_size
            user_batch = test_users[start:end]
            scores = self.predict(user_batch)
            for idx,user in enumerate(user_batch):
                items = data_generator.train_items.get(user, None)
                user_pos_test = test_set[user]
                user_score = scores[idx]
                if items is not None:
                    user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, k=max(Ks), dim=-1)
                topk_idx = topk_idx.cpu().numpy()  

                gt_items = np.asarray(user_pos_test, dtype=np.int64)
                mask = np.isin(topk_idx, gt_items, assume_unique=False)
                topk_list.append(mask.astype(np.int8))
            
        assert len(topk_list) == n_test_users
        topk_list = np.array(topk_list)
        metric_dict = calculate_result(topk_list, gt_length)
        return metric_dict

def calculate_result(topk_list, gt_len):
    metrics = ["recall", "ndcg"]
    result_list = []
    for metric in metrics:
        metric_fuc = metrics_dict[metric.lower()]
        result = metric_fuc(topk_list, gt_len)
        result_list.append(result)
    result_list = np.stack(result_list, axis=0).mean(axis=1)
    metric_dict = {}
    for metric in metrics:
        metric_dict[metric] = []

    for topk in Ks:
        for metric, value in zip(metrics, result_list):
            metric_dict[metric].append(np.round(value[topk - 1], 5))

    return metric_dict

def bpr_loss(users, pos_items, neg_items):
    pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
    neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

    maxi = F.logsigmoid(pos_scores - neg_scores)
    mf_loss = -torch.mean(maxi)

    return mf_loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PYTHONHASHSEED"] = str(seed)

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(2025)

    config = dict()
    config["device"] = device
    config["n_users"] = data_generator.n_users
    config["n_items"] = data_generator.n_items
    config["behs"] = data_generator.behs
    config["trn_mat"] = data_generator.trnMats[-1]

    n_users, n_items = data_generator.n_users, data_generator.n_items
    behs = data_generator.behs
    n_behs = data_generator.beh_num

    trnDicts = copy.deepcopy(data_generator.trnDicts)
    max_item_list = []
    beh_label_list = []
    for i in range(n_behs):
        max_item, beh_label = get_lables(trnDicts[i])
        max_item_list.append(max_item)
        beh_label_list.append(beh_label)

    t0 = time()
    model = CIDER(args, device, max_item_list).to(device)

    user_train1, beh_item_list = get_train_instances1(max_item_list, beh_label_list)
    n_batch = int(len(user_train1) / args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',
        factor=0.5,
        patience=2,
    )
    
    cur_best_pre_0 = 0.0
    run_time = datetime.strftime(datetime.now(), "%Y_%m_%d__%H_%M_%S")
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []
    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):
        model.train()
        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        beh_item_list = [beh_item[shuffle_indices] for beh_item in beh_item_list]
        total_denoise, total_cpt, total_bpr, total_loss = 0.0, 0.0, 0.0, 0.0
        t1 = time()
        for idx in range(n_batch):
            optimizer.zero_grad()
            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            beh_batch = [
                beh_item[start_index:end_index] for beh_item in beh_item_list
            ]  # [[B, max_item1], [B, max_item2], [B, max_item3]]

            beh_sample = []
            for beh in range(n_behs):
                pos_items = []
                neg_items = []
                for u in u_batch:
                    pos, neg = data_generator.sample_behvaior_item(u, beh, 1)
                    pos_items += pos
                    neg_items += neg

                beh_sample.append([pos_items, neg_items])
            
            # load into cuda
            u_batch = torch.from_numpy(u_batch).to(device)
            beh_batch = [torch.from_numpy(beh_item).to(device) for beh_item in beh_batch]
            with autocast('cuda'):
                denoise_loss, cpt_loss, bpr = model(u_batch, beh_batch, beh_sample)
                batch_loss = denoise_loss + cpt_loss + bpr
            scaler.scale(batch_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            total_denoise += denoise_loss.item() / n_batch
            total_cpt += cpt_loss.item() / n_batch
            total_bpr += bpr.item() / n_batch
            total_loss += batch_loss.item() / n_batch
        if np.isnan(total_loss) == True:
            sys.exit()

        if (epoch + 1) % args.test_epoch != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = "Epoch %d [%0.1fs]: train[%.5f] == [%.5f + %.5f + %.5f]" % (
                    epoch,
                    time() - t1,
                    total_loss,
                    total_denoise,
                    total_cpt,
                    total_bpr,
                )
            print(perf_str)
            continue

        t2 = time()
        model.eval()
        with torch.no_grad():
            users_to_test = list(data_generator.test_set.keys())
            ret = model.test_torch(users_to_test)

        t3 = time()

        loss_loger.append(total_loss)
        rec_loger.append(ret["recall"])
        ndcg_loger.append(ret["ndcg"])

        if args.verbose > 0:
            perf_str = (
                "Epoch %d [%.1fs + %.1fs]:, recall=[%.5f, %.5f, %.5f], "
                " ndcg=[%.5f, %.5f, %.5f]"
                % (
                    epoch,
                    t2 - t1,
                    t3 - t2,
                    ret["recall"][0],
                    ret["recall"][1],
                    ret["recall"][2],
                    ret["ndcg"][0],
                    ret["ndcg"][1],
                    ret["ndcg"][2],
                )
            )
            print(datetime.now().strftime("%Y-%m-%d %H:%M: "), perf_str)

        cur_best_pre_0, stopping_step, should_stop, flag = early_stopping_new(
            ret["recall"][0], cur_best_pre_0, stopping_step, expected_order="acc", flag_step=10
        )
        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break
        if args.lr_decay: 
            scheduler.step(ret["recall"][0]) 

    recs = np.array(rec_loger)
    ndcgs = np.array(ndcg_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], ndcg=[%s]" % (
        idx,
        time() - t0,
        "\t".join(["%.4f" % r for r in recs[idx]]),
        "\t".join(["%.4f" % r for r in ndcgs[idx]]),
    )
    print(datetime.now().strftime("%Y-%m-%d %H:%M: "), final_perf)
