
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from author_disambiguation.utils import construct_doc_matrix


def compute_f1(ground_truth_dict, bpr_optimizer):
    """
    perform Hierarchy Clustering on doc embedding matrix
    for name disambiguation
    use cluster-level mean F1 for evaluation
    """
    work_list = list(ground_truth_dict.keys())
    D_matrix = construct_doc_matrix(bpr_optimizer.work_latent_matrix, work_list)

    true_clusters = set()
    for k,auths in ground_truth_dict.items():
        true_clusters = true_clusters.union(set(auths))
    true_cluster_size = len(true_clusters)
    if true_cluster_size > D_matrix.shape[0]:
        print('True cluster size: {}, Approximated cluster size: {}'.format(true_cluster_size, D_matrix.shape[0]))
        return None
    y_pred = AgglomerativeClustering(n_clusters = true_cluster_size,
                                     linkage = "average",
                                     affinity = "cosine").fit_predict(D_matrix)

    true_label_dict = {}

    for i in range(len(work_list)):
        authors = ground_truth_dict[work_list[i]]
        for true_lbl in authors:
            if true_lbl not in true_label_dict:
                true_label_dict[true_lbl] = [i]
            else:
                true_label_dict[true_lbl].append(i)

    predict_label_dict = {}
    for idx, pred_lbl in enumerate(y_pred):
        if pred_lbl not in predict_label_dict:
            predict_label_dict[pred_lbl] = [idx]
        else:
            predict_label_dict[pred_lbl].append(idx)

    # compute cluster-level F1
    # let's denote C(r) as clustering result and T(k) as partition (ground-truth)
    # construct r * k contingency table for clustering purpose
    r_k_table = []
    for _,v1 in predict_label_dict.items():
        k_list = []
        for _,v2 in true_label_dict.items():
            N_ij = len(set(v1).intersection(v2))
            k_list.append(N_ij)
        r_k_table.append(k_list)
    r_k_matrix = np.array(r_k_table)
    r_num = int(r_k_matrix.shape[0])

    # compute F1 for each row C_i
    sum_f1 = 0.0
    for row in range(r_num):
        row_sum = np.sum(r_k_matrix[row,:])
        if row_sum != 0:
            max_col_index = np.argmax(r_k_matrix[row,:])
            row_max_value = r_k_matrix[row, max_col_index]
            prec = float(row_max_value) / row_sum
            col_sum = np.sum(r_k_matrix[:, max_col_index])
            rec = float(row_max_value) / col_sum
            row_f1 = float(2 * prec * rec) / (prec + rec)
            sum_f1 += row_f1

    average_f1 = float(sum_f1) / r_num
    return average_f1
