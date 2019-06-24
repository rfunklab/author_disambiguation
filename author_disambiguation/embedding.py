# Optimizer from https://arxiv.org/abs/1702.02287

import numpy as np
from author_disambiguation.utils import sigmoid
import progressbar



class BprOptimizer():
    """
    Bayesian Personalized Ranking for objective loss
    latent_dim: latent dimension
    alpha: learning rate
    matrix_reg: regularization parameter of matrix
    """
    def __init__(self, latent_dim, alpha, matrix_reg):
        self.latent_dim = latent_dim
        self.alpha = alpha
        self.matrix_reg = matrix_reg

    def init_model(self, coauthor_graph, work_graph):
        """
        initialize matrix using uniform [-0.2, 0.2]
        """
        self.work_latent_matrix = {}
        self.author_latent_matrix = {}
        works = list(work_graph.nodes())
        for work_idx in works:
            self.work_latent_matrix[work_idx] = np.random.uniform(-0.2, 0.2,
                                                                    self.latent_dim)
        coauthors = list(coauthor_graph.nodes())
        for author_idx in coauthors:
            self.author_latent_matrix[author_idx] = np.random.uniform(-0.2, 0.2,
                                                                      self.latent_dim)

    def update_pp_gradient(self, fst, snd, third):
        """
        SGD inference
        """
        x = self.predict_score(fst, snd, "pp") - \
            self.predict_score(fst, third, "pp")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.author_latent_matrix[snd] - \
                                  self.author_latent_matrix[third]) + \
                    2 * self.matrix_reg * self.author_latent_matrix[fst]
        self.author_latent_matrix[fst] = self.author_latent_matrix[fst] - \
                                         self.alpha * grad_fst

        grad_snd = common_term * self.author_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.author_latent_matrix[snd]
        self.author_latent_matrix[snd]= self.author_latent_matrix[snd] - \
                                        self.alpha * grad_snd

        grad_third = -common_term * self.author_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.author_latent_matrix[third]
        self.author_latent_matrix[third] = self.author_latent_matrix[third] - \
                                           self.alpha * grad_third

    def update_pd_gradient(self, fst, snd, third):
        x = self.predict_score(fst, snd, "pd") - \
            self.predict_score(fst, third, "pd")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.author_latent_matrix[snd] - \
                                  self.author_latent_matrix[third]) + \
                   2 * self.matrix_reg * self.work_latent_matrix[fst]
        self.work_latent_matrix[fst] = self.work_latent_matrix[fst] - \
                                         self.alpha * grad_fst

        grad_snd = common_term * self.work_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.author_latent_matrix[snd]
        self.author_latent_matrix[snd]= self.author_latent_matrix[snd] - \
                                        self.alpha * grad_snd

        grad_third = -common_term * self.work_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.author_latent_matrix[third]
        self.author_latent_matrix[third] = self.author_latent_matrix[third] - \
                                           self.alpha * grad_third

    def update_dd_gradient(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dd") - \
            self.predict_score(fst, third, "dd")
        common_term = sigmoid(x) - 1

        grad_fst = common_term * (self.work_latent_matrix[snd] - \
                                  self.work_latent_matrix[third]) + \
                   2 * self.matrix_reg * self.work_latent_matrix[fst]
        self.work_latent_matrix[fst] = self.work_latent_matrix[fst] - \
                                         self.alpha * grad_fst

        grad_snd = common_term * self.work_latent_matrix[fst] + \
                   2 * self.matrix_reg * self.work_latent_matrix[snd]
        self.work_latent_matrix[snd]= self.work_latent_matrix[snd] - \
                                       self.alpha * grad_snd

        grad_third = -common_term * self.work_latent_matrix[fst] + \
                     2 * self.matrix_reg * self.work_latent_matrix[third]
        self.work_latent_matrix[third] = self.work_latent_matrix[third] - \
                                          self.alpha * grad_third

    def compute_pp_loss(self, fst, snd, third):
        """
        loss includes ranking loss and model complexity
        """
        x = self.predict_score(fst, snd, "pp") - \
             self.predict_score(fst, third, "pp")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[fst],
                                               self.author_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[snd],
                                               self.author_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[third],
                                               self.author_latent_matrix[third])
        return ranking_loss + complexity

    def compute_pd_loss(self, fst, snd, third):
        x = self.predict_score(fst, snd, "pd") - \
            self.predict_score(fst, third, "pd")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.work_latent_matrix[fst],
                                               self.work_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[snd],
                                               self.author_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.author_latent_matrix[third],
                                               self.author_latent_matrix[third])
        return ranking_loss + complexity

    def compute_dd_loss(self, fst, snd, third):
        x = self.predict_score(fst, snd, "dd") - \
            self.predict_score(fst, third, "dd")
        ranking_loss = -np.log(sigmoid(x))

        complexity = 0.0
        complexity += self.matrix_reg * np.dot(self.work_latent_matrix[fst],
                                               self.work_latent_matrix[fst])
        complexity += self.matrix_reg * np.dot(self.work_latent_matrix[snd],
                                               self.work_latent_matrix[snd])
        complexity += self.matrix_reg * np.dot(self.work_latent_matrix[third],
                                               self.work_latent_matrix[third])
        return ranking_loss + complexity

    def predict_score(self, fst, snd, graph_type):
        """
        pp: person-person network
        pd: person-document bipartite network
        dd: doc-doc network
        detailed notation is inside paper
        """
        if graph_type == "pp":
            return np.dot(self.author_latent_matrix[fst],
                          self.author_latent_matrix[snd])
        elif graph_type == "pd":
            return np.dot(self.work_latent_matrix[fst],
                          self.author_latent_matrix[snd])
        elif graph_type == "dd":
            return np.dot(self.work_latent_matrix[fst],
                          self.work_latent_matrix[snd])


def embed(num_epoch, coauthor_graph, author_work_graph, work_graph,
        bpr_optimizer, pp_sampler, pd_sampler, dd_sampler, ground_truth_dict,
        evaluation, sampler_method='uniform'):

    num_nnz = work_graph.number_of_edges() + \
            coauthor_graph.number_of_edges() + \
            author_work_graph.number_of_edges()
    bpr_optimizer.init_model(coauthor_graph, work_graph)

    if sampler_method == 'uniform':
        for epoch in range(num_epoch):
            print('Epoch: {}'.format(epoch))
            bar = progressbar.ProgressBar(maxval=num_nnz, \
            widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()

            bpr_loss = 0.0
            for bu in range(num_nnz):
                """
                update embedding in person-person network
                update embedding in person-document network
                update embedding in doc-doc network
                """
                bar.update(bu)
                for i, j, t in pp_sampler.generate_triplet_uniform(coauthor_graph):
                    bpr_optimizer.update_pp_gradient(i, j, t)
                    bpr_loss += bpr_optimizer.compute_pp_loss(i, j, t)

                for i, j, t in pd_sampler.generate_triplet_uniform(author_work_graph, coauthor_graph, work_graph):
                    bpr_optimizer.update_pd_gradient(i, j, t)
                    bpr_loss += bpr_optimizer.compute_pd_loss(i, j, t)

                for i, j, t in dd_sampler.generate_triplet_uniform(work_graph):
                    bpr_optimizer.update_dd_gradient(i, j, t)
                    bpr_loss += bpr_optimizer.compute_dd_loss(i, j, t)
            bar.finish()
            average_loss = float(bpr_loss) / num_nnz
            print('\n Average BPR loss: {}'.format(average_loss))
            average_f1 = evaluation(ground_truth_dict, bpr_optimizer)
            print('F1: {} \n'.format(average_f1))


    elif sampler_method == 'reject':
        for _ in range(num_epoch):
            #bpr_loss = 0.0
            for _ in xrange(0, num_nnz):
                """
                update embedding in person-person network
                update embedding in person-document network
                update embedding in doc-doc network
                """
                for i, j, t in pp_sampler.generate_triplet_reject(dataset, bpr_optimizer):
                    bpr_optimizer.update_pp_gradient(i, j, t)
                    #bpr_loss += bpr_optimizer.compute_pp_loss(i, j, t)

                for i, j, t in pd_sampler.generate_triplet_reject(dataset, bpr_optimizer):
                    bpr_optimizer.update_pd_gradient(i, j, t)
                    #bpr_loss += bpr_optimizer.compute_pd_loss(i, j, t)

                for i, j, t in dd_sampler.generate_triplet_reject(dataset, bpr_optimizer):
                    bpr_optimizer.update_dd_gradient(i, j, t)
                    #bpr_loss += bpr_optimizer.compute_dd_loss(i, j, t)

    elif sampler_method == 'adaptive':
        for _ in xrange(0, num_epoch):
            #bpr_loss = 0.0
            for _ in xrange(0, num_nnz):
                """
                update embedding in person-person network
                update embedding in person-document network
                update embedding in doc-doc network
                """
                for i, j, t in pp_sampler.generate_triplet_adaptive(dataset, bpr_optimizer):
                    bpr_optimizer.update_pp_gradient(i, j, t)
                    #bpr_loss += bpr_optimizer.compute_pp_loss(i, j, t)

                for i, j, t in pd_sampler.generate_triplet_adaptive(dataset, bpr_optimizer):
                    bpr_optimizer.update_pd_gradient(i, j, t)
                    #bpr_loss += bpr_optimizer.compute_pd_loss(i, j, t)

                for i, j, t in dd_sampler.generate_triplet_adaptive(dataset, bpr_optimizer):
                    bpr_optimizer.update_dd_gradient(i, j, t)
                    #bpr_loss += bpr_optimizer.compute_dd_loss(i, j, t)

    # save_embedding(bpr_optimizer.paper_latent_matrix,
    #                dataset.paper_list, bpr_optimizer.latent_dimen)
