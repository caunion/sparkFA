import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import FactorAnalysis
from PPCA import PPCA

def data2cov(data):
    [n, d] = data.shape
    mu = np.mean(data, 0)
    mu = np.tile(mu, [n, 1])
    sigma = np.var(data, 0)
    sigma = np.tile(sigma, [n, 1])
    norm_data = (1.0/n) * (data - mu ) / (sigma)
    covar = norm_data.transpose().dot(norm_data)
    return covar, norm_data

def data2norm(data):
    [n, d] = data.shape
    mu = np.mean(data, 0)
    mu = np.tile(mu, [n, 1])
    sigma = np.var(data, 0)
    sigma = np.tile(sigma, [n, 1])
    norm_data = (1.0/n) * (data - mu ) / (sigma)
    return norm_data

def gen(shape, rank, noise_type = 0):
    n, d = shape
    assert rank <= min(shape)
    if len(shape) == 1:
        h = w = shape
    else:
        h, w = shape
    A = (np.random.rand(h, rank))* 10.0 / rank
    B = np.random.rand(rank, w)* 1.0
    C = A.dot(B)

    noise = np.zeros([n, d])

    mag = np.std(C, 0)
    a = np.max(mag) * 0.01
    b = np.max(mag) * 0.3

    if noise_type == 3:
        a = np.max(mag) * 0.5
        b = np.max(mag) * 1.0
    scale = ( np.random.rand() * a + b)

    # sigma = (np.random.rand(1,d)) * 5 * scale
    sigma = abs(  (np.random.randn(1, d)) * (2*scale) + scale )
    for i in range(n):
        if noise_type == 1:
            noise_single = (np.random.randn(1, d)) * scale
            noise_single = noise_single.flatten()
        elif noise_type == 2:
            noise_single = np.random.randn(1, d) * sigma
            noise_single = noise_single.flatten()
        elif noise_type == 3:
            noise_single = np.random.randn(1, d) * sigma * 2.0
            noise_single = noise_single.flatten()
        else:
            break
        noise[i, :] = noise_single

    high_rank = C + noise

    low_rank = A
    mapper = B

    return high_rank, low_rank, noise


def pca(data, dim):
    norm_data = data2norm(data)
    [u, s , v] = np.linalg.svd(norm_data)
    # [w, v] = np.linalg.eig(covar)
    # I = sorted(range(len(w)), key = lambda k : w[k], reverse=True)
    # A = w[I]
    # Q = v[I,:]

    # low_expression = norm_data.dot(u[:, 0:dim])
    variablity = np.sum(s[range(dim)]) / np.sum(s)
    return variablity


def simulate():
    shape =[1000, 7]
    n, d= shape
    rank = 3
    data_pure, low_pure, noise_pure = gen(shape, rank, noise_type=0)
    data_sig, low_sig, noise_sig = gen(shape, rank, noise_type=1)
    data_diff, low_diff, noise_diff= gen(shape, rank, noise_type=2)
    data_diff2, low_diff2, noise_diff2 = gen(shape, rank, noise_type=3)

    np.savetxt("data_pure.txt", data_pure)
    np.savetxt("data_sig.txt", data_sig)
    np.savetxt("data_diff_easy.txt", data_diff)
    np.savetxt("data_diff_hard.txt", data_diff2)


    var_pure= pca(data_pure, rank)
    var_sig = pca(data_sig, rank)
    var_diff = pca(data_diff, rank)
    var_diff2 = pca(data_diff2, rank)


    ppca= PPCA(data_pure)
    ppca.fit(d=d, verbose=True)
    var_pure_ppca =np.sum(ppca.eig_vals[range(rank)])/ np.sum(ppca.eig_vals)

    ppca= PPCA(data_sig)
    ppca.fit(d=d, verbose=True)
    var_sig_ppca =np.sum(abs(ppca.eig_vals[range(rank)]))/ np.sum(abs(ppca.eig_vals))

    ppca= PPCA(data_diff)
    ppca.fit(d=d, verbose=True)
    var_diff_ppca =np.sum(abs(ppca.eig_vals[range(rank)]))/ np.sum(abs(ppca.eig_vals))

    ppca= PPCA(data_diff2)
    ppca.fit(d=d, verbose=True)
    var_diff2_ppca =np.sum(abs(ppca.eig_vals[range(rank)]))/ np.sum(abs(ppca.eig_vals))

    fa = FactorAnalysis()

    var_pca = [var_pure, var_sig, var_diff, var_diff2]
    var_ppca = [var_pure_ppca, var_sig_ppca, var_diff_ppca, var_diff2_ppca]
    return var_pca, var_ppca


def gen_classification(shape, n_rank=3, n_classes=4, noise_type = 0):
    from sklearn.datasets import make_classification
    from mpl_toolkits.mplot3d import Axes3D
    from sklearn.utils import shuffle
    from sklearn.utils import random
    from sklearn.svm import SVC
    n_sample, n_dim = shape
    colors = ['r','g','b','y']
    X, y = make_classification(n_samples=n_sample,
                               n_features=n_rank,
                               n_repeated=0,
                               n_redundant=0,
                               n_clusters_per_class=1,
                               n_informative=n_rank,
                               n_classes=n_classes,
                               random_state=1)
    # fig = plt.figure()
    # ax =fig.add_subplot(111, projection='3d')
    # # ax =fig.add_subplot(111)
    # for i in range(n_classes):
    #     I = (y == i)
    #     x = X[I, :]
    #     ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colors[i])
    #     # ax.scatter(x[:, 0], x[:, 1], c=colors[i])
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    clf = SVC(C=1)
    clf.fit(X, y)
    init_acc = clf.score(X, y)

    transform = np.random.rand(n_rank, n_dim)

    X_high = X.dot(transform)
    mag = np.std(X_high, 0)


    if noise_type == 0:
        sigma = 0 # no noise
    if noise_type == 1:
        a = 0.1 * mag.mean()
        b = 0.3 * mag.mean()
        sigma = np.random.rand()
    if noise_type == 2:
        a = 0.1 * mag
        b = 0.3 * mag
        sigma = np.random.rand(1, n_dim) * (b - a) + a
    if noise_type == 3:
        a = 0.5 * mag
        b = 1.0 * mag
        sigma = np.random.rand(1, n_dim) * (b - a) + a

    mu = np.tile(np.random.randn(1, n_dim), [n_sample, 1])

    # noise = np.tile(np.random.randn(1, n_dim) * sigma, [n_sample, 1])

    noise = np.zeros([n_sample, n_dim])
    for i in range(n_sample):
        noise[i, :] = np.random.randn(1, n_dim) * sigma

    X_noise = X_high + noise + mu

    return X_noise, init_acc, X, y


def classification_sim():
    shape = [1000, 10]
    n_rank = 3
    n_class = 5
    C = 2
    X_pure, _, _,_ = gen_classification(shape=shape, n_rank=n_rank, n_classes=n_class, noise_type = 0)
    X_sig, _, _, _ = gen_classification(shape=shape, n_rank=n_rank, n_classes=n_class, noise_type = 1)
    X_lit, _, _, _ = gen_classification(shape=shape, n_rank=n_rank, n_classes=n_class, noise_type = 2)
    X_hvy, init_acc, X, y = gen_classification(shape=shape, n_rank=n_rank, n_classes=n_class, noise_type=2)

    np.savetxt("X_pure.txt", X_pure)
    np.savetxt("X_sig.txt", X_sig)
    np.savetxt("X_lit.txt", X_lit)
    np.savetxt("X_hvy.txt", X_hvy)

    from sklearn.svm import SVC
    from sklearn.decomposition import PCA, FactorAnalysis

    pca = PCA(n_components=n_rank)
    pca.fit(X_pure)
    X_low = pca.transform(X_pure)
    clf = SVC(C=C)
    clf.fit(X_low, y)
    acc_pure_pca = clf.score(X_low, y)
    print "acc_pure_pca: %.4f" % acc_pure_pca

    pca = PCA(n_components=n_rank)
    pca.fit(X_sig)
    X_low = pca.transform(X_sig)
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_low, y)
    acc_sig_pca = clf.score(X_low, y)
    print "acc_sig_pca: %.4f" % acc_sig_pca

    pca = PCA(n_components=n_rank)
    pca.fit(X_lit)
    X_low = pca.transform(X_lit)
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_low, y)
    acc_lit_pca = clf.score(X_low, y)
    print "acc_lit_pca: %.4f" % acc_lit_pca

    pca = PCA(n_components=n_rank)
    pca.fit(X_hvy)
    X_low = pca.transform(X_hvy)
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_low, y)
    acc_hvy_pca = clf.score(X_low, y)
    print "acc_hvy_pca: %.4f" % acc_hvy_pca

    pca = None

    # PPCA
    ppca = PPCA(X_pure)
    ppca.fit(d=n_rank)
    X_low = ppca.transform(X_pure)
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_low, y)
    acc_pure_ppca = clf.score(X_low, y)
    print "acc_pure_ppca: %.4f" % acc_pure_ppca

    ppca = PPCA(X_sig)
    ppca.fit(d=n_rank)
    X_low = ppca.transform(X_sig)
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_low, y)
    acc_sig_ppca = clf.score(X_low, y)
    print "acc_sig_ppca: %.4f" % acc_sig_ppca

    ppca = PPCA(X_lit)
    ppca.fit(d=n_rank)
    X_low = ppca.transform(X_lit)
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_low, y)
    acc_lit_ppca = clf.score(X_low, y)
    print "acc_lit_ppca: %.4f" % acc_lit_ppca

    ppca = PPCA(X_hvy)
    ppca.fit(d=n_rank)
    X_low = ppca.transform(X_hvy)
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_low, y)
    acc_hvy_ppca = clf.score(X_low, y)
    print "acc_hvy_ppca: %.4f" % acc_hvy_ppca

    ## FA
    fa = FactorAnalysis(n_components=n_rank)
    fa.fit(X_pure)
    X_low = fa.transform(X_pure)
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_low, y)
    acc_pure_fa = clf.score(X_low, y)
    print "acc_pure_fa: %.4f" % acc_pure_fa

    fa = FactorAnalysis(n_components=n_rank)
    fa.fit(X_sig)
    X_low = fa.transform(X_sig)
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_low, y)
    acc_sig_fa = clf.score(X_low, y)
    print "acc_sig_fa: %.4f" % acc_sig_fa

    fa = FactorAnalysis(n_components=n_rank)
    fa.fit(X_lit)
    X_low = fa.transform(X_lit)
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_low, y)
    acc_lit_fa = clf.score(X_low, y)
    print "acc_lit_fa: %.4f" % acc_lit_fa

    fa = FactorAnalysis(n_components=n_rank)
    fa.fit(X_hvy)
    X_low = fa.transform(X_hvy)
    clf = SVC(C=C, kernel='linear')
    clf.fit(X_low, y)
    acc_hvy_fa = clf.score(X_low, y)
    print "acc_hvy_fa: %.4f" % acc_hvy_fa


def main():
    n = 10
    groups = 4
    N = np.arange(groups)
    vars_ppca = np.zeros([n, groups])
    vars_pca = np.zeros([n, groups])
    for i in range(n):
        vars_pca[i, :], vars_ppca[i, :] = simulate()

    mean_var_pca = np.mean(vars_pca, 0)
    mean_var_ppca = np.mean(vars_ppca, 0)
    std_var_pca = np.std(vars_pca, 0)
    std_var_ppca = np.std(vars_ppca, 0)
    fig, ax = plt.subplots()
    width = 0.2
    rect1 = ax.bar(N, mean_var_pca, width, color= 'r', yerr=std_var_pca)
    rect2 = ax.bar(N+width, mean_var_ppca, width, color = 'y', yerr=std_var_ppca)

    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]

    ax.set_ylim([0.3, 1.1])
    ax.set_ylabel('Variability')
    ax.set_title('Variability of different PCA vs PPCA')
    ax.set_xticks(N + width)
    ax.set_xticklabels(['$\\boldsymbol{X}^{(1)}$', '$\\boldsymbol{X}^{(2)}$', '$\\boldsymbol{X}^{(3)}$', '$\\boldsymbol{X}^{(4)}$'])

    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2., 1.02 * height,
                    '%.4f' % height,
                    ha='center', va='bottom')

    autolabel(rect1)
    autolabel(rect2)
    ax.legend((rect1[0], rect2[0]), ('PCA', 'PPCA'))
    plt.show()

if __name__ == "__main__":
    classification_sim()
