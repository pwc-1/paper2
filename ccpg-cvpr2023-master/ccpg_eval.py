# This source is based on https://github.com/ShiqiYu/OpenGait/blob/master/opengait/evaluation/evaluator.py
# These codes are used for evaluation on CCPG dataset.
# These codes are constructed for MindSpore, a deep learning architecture.

import numpy as np
# np.set_printoptions(threshold=np.inf)

def cuda_dist(x, y, metric='euc'):
    num_bin = x.shape[2]
    n_x = x.shape[0]
    n_y = y.shape[0]
    dist = np.zeros((n_x, n_y))
    for i in range(num_bin):
        _x = x[:, :, i]
        _y = y[:, :, i]
        if metric == 'cos':
            _x = _x / np.linalg.norm(_x, axis=1, keepdims=True)
            _y = _y / np.linalg.norm(_y, axis=1, keepdims=True)
            dist += np.dot(_x, _y.T)
        else:
            # _dist = np.sum(_x ** 2, axis=1, keepdims=True) + np.sum(_y ** 2, axis=1, keepdims=True).T - 2 * np.dot(_x, _y.T)
            _x = _x.astype(np.float64)
            _y = _y.astype(np.float64)
            _dist = np.sum(_x ** 2, axis=1, keepdims=True) + np.sum(_y ** 2, axis=1, keepdims=True).T - 2 * np.dot(_x, _y.T)
            dist += np.sqrt(np.maximum(_dist, 0))
    return 1 - dist/num_bin if metric == 'cos' else dist / num_bin

def rank_cmc(ipt):
        new = []
        for idx, i in enumerate(ipt):
            print(type(ipt[idx]), ipt[idx].shape)
            if idx == 1:
                new.append(ipt[idx])
            else:
                new.append(ipt[idx][::-1])
        return new

# Define CCPG Evaluation

def compute_evaluate_ccpg(distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50):
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    all_INP = []
    num_valid_q = 0.
    for q_idx in range(num_q):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        orig_cmc = matches[q_idx][keep] # binary vector, positions with value 1 are correct matches
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()

        pos_idx = np.where(orig_cmc == 1)
        max_pos_idx = np.max(pos_idx)
        inp = cmc[max_pos_idx]/ (max_pos_idx + 1.0)
        all_INP.append(inp)

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i+1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)
    mINP = np.mean(all_INP)

    return all_cmc, mAP#, mINP


def CCPG_EVAL(data, dataset, metric='euc'):
    feature, label, seq_type, view = data['embeddings'], data['labels'], data['types'], data['views']

    label = np.array(label)
    for i in range(len(view)):
        view[i] = view[i].split("_")[0]
    view_np = np.array(view)
    view_list = list(set(view))
    view_list.sort()
    view_num = len(view_list)

    probe_seq_dict = {'CCPG': [["U0_D0_BG", "U0_D0"], ["U3_D3"], ["U1_D0"]]}

    gallery_seq_dict = {'CCPG':[["U1_D1", "U2_D2", "U3_D3"], ["U0_D3"], ["U1_D1"]]}
    if dataset not in (probe_seq_dict or gallery_seq_dict):
        raise KeyError("DataSet %s hasn't been supported !" % dataset)

    cmc_save = []
    ap_save = []
    for (p, probe_seq) in enumerate(probe_seq_dict[dataset]):
        gallery_seq = gallery_seq_dict[dataset][p]
        gseq_mask = np.isin(seq_type, gallery_seq)

        gallery_x = feature[gseq_mask, :, :]
        gallery_y = label[gseq_mask]
        gallery_view = view_np[gseq_mask]
        
        pseq_mask = np.isin(seq_type, probe_seq)
        probe_x = feature[pseq_mask, :, :]
        probe_y = label[pseq_mask]
        probe_view = view_np[pseq_mask]

        distmat = np.asarray(cuda_dist(probe_x, gallery_x, metric))

        cmc, ap = compute_evaluate_ccpg(distmat, probe_y, gallery_y, probe_view, gallery_view)
        cmc_save.append(cmc)
        ap_save.append(ap)
        

    # cmc_save = rank_cmc(cmc_save)
    print(cmc_save)
    print(ap_save)
    print(
            '===Rank-1 (Exclude identical-view cases for Person Re-Identification)===')
    print('CL: %.3f,\tUP: %.3f,\tDN: %.3f' % (cmc_save[0][0]*100, cmc_save[1][0]*100, cmc_save[2][0]*100))

    print(
            '===mAP (Exclude identical-view cases for Person Re-Identification)===')
    print('CL: %.3f,\tUP: %.3f,\tDN: %.3f' % (ap_save[0]*100, ap_save[1]*100, ap_save[2]*100))




if __name__ == "__main__":
    # Load data 1
    embeddings = np.load("/home/liweijia/run_mind/OpenGait/feature.npy")
    id_label = np.load("/home/liweijia/run_mind/OpenGait/label.npy").tolist()
    cloth_label = np.load("/home/liweijia/run_mind/OpenGait/seq_type.npy").tolist()
    view_label = np.load("/home/liweijia/run_mind/OpenGait/view.npy").tolist()
    input_data = {'embeddings': embeddings, 
              'labels': id_label,
              'types': cloth_label,
              'views': view_label}
    
    # Load data 2
    # import pickle
    # with open("data.pkl", "rb") as f:
    #     ipts = pickle.load(f)
    # input_data = {'embeddings': ipts.item()['embeddings'], 
    #           'labels': ipts.item()['labels'],
    #           'types': ipts.item()['types'],
    #           'views': ipts.item()['views']}
    
    CCPG_EVAL(input_data, "CCPG")