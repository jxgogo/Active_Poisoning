import numpy as np
from scipy.spatial.distance import pdist, squareform, cdist
from sklearn.cluster import KMeans
from tensorflow.keras.models import Model
from lib import utils


def Entropy(x_p, y_p, batch_size, model, isdeep=True, active=False):
    if isdeep:
        y_pred_pro = model.predict(x_p)
    else:
        y_pred_pro = model.predict_proba(x_p)
    idx1 = np.argwhere(y_p != 1)
    idx1 = idx1.squeeze()

    ent = np.sum(-y_pred_pro[idx1] * np.log(y_pred_pro[idx1] + 1e-8), axis=1)
    assert (len(np.shape(ent)) == 1)
    if active:
        idx2 = np.argsort(ent)[-batch_size:]
    else:
        idx2 = np.argsort(ent)[:batch_size]
    return idx1[idx2]


def MaxModelChange(x_p, y_p, batch_size, model, isdeep=True, active=False):
    if isdeep:
        y_pred_pro = model.predict(x_p)
        x_feature_layer = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
        x_feature = x_feature_layer.predict(x_p)
    else:
        y_pred_pro = model.predict_proba(x_p)
        x_feature = x_p

    idx1 = np.argwhere(y_p != 1)
    idx1 = idx1.squeeze()
    y_pred_pro = y_pred_pro[idx1]
    x_feature = x_feature[idx1]
    MMC = y_pred_pro[:, 1] * np.linalg.norm(x_feature, axis=1)

    if active:
        idx2 = np.argsort(MMC)[-batch_size:]
    else:
        idx2 = np.argsort(MMC)[:batch_size]
    return idx1[idx2]


def RDS(x_p, y_p, batch_size, model=None, isdeep=True):
    if isdeep:
        x_feature_layer = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
        x_feature = x_feature_layer.predict(x_p)
    else:
        x_feature = x_p
    idx1 = np.argwhere(y_p != 1)
    idx1 = idx1.squeeze()
    x_feature = x_feature[idx1]
    # distX = squareform(pdist(x_feature))
    # RD initialization
    kmeans = KMeans(n_clusters=batch_size, random_state=0).fit(x_feature)
    idsCluster = kmeans.labels_
    centers = kmeans.cluster_centers_
    # Dist = np.zeros((idsCluster.size, centers.shape[0]))
    # for i in range(idsCluster.size):
    #     for j in range(centers.shape[0]):
    #         Dist[i, j] = np.linalg.norm(x_feature[i, :] - centers[j, :])
    Dist = cdist(x_feature, centers)
    idx2 = np.zeros(batch_size, dtype=int)
    for n in range(batch_size):
        idsP_n = np.argwhere(idsCluster == n).reshape([-1])
        idx2[n] = idsP_n[np.argmin(Dist[idsP_n, n])]
    return idx1[idx2]


def Random(x_p, y_p, batch_size):
    idx1 = np.argwhere(y_p != 1)
    idx1 = idx1.squeeze()
    idx2 = utils.shuffle_data(len(x_p[idx1]))
    return idx1[idx2[:batch_size]]


def MDS(x_p, y_p, batch_size, model=None, isdeep=True):
    if isdeep:
        x_feature_layer = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
        x_feature = x_feature_layer.predict(x_p)
    else:
        x_feature = x_p
    idx1 = np.argwhere(y_p != 1)
    idx1 = idx1.squeeze()
    x = x_feature[idx1]
    distX = squareform(pdist(x))
    i = np.arange(x.shape[0])
    distX[i, i] = 999
    idx2 = np.argsort(np.min(distX, axis=1))
    return idx1[idx2[-batch_size:]]


def MUS_MDS(x_p, y_p, batch_size, model, isdeep=True, active=False, weight=0.5):
    if isdeep:
        y_pred_pro = model.predict(x_p)
        x_feature_layer = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
        x_feature = x_feature_layer.predict(x_p)
    else:
        y_pred_pro = model.predict_proba(x_p)
        x_feature = x_p
    idx1 = np.argwhere(y_p != 1)
    idx1 = idx1.squeeze()
    ent = np.sum(-y_pred_pro[idx1] * np.log(y_pred_pro[idx1] + 1e-8), axis=1)
    ent = (ent - np.min(ent)) / (np.max(ent) - np.min(ent)) * weight

    x = x_feature[idx1]
    distX = squareform(pdist(x))
    i = np.arange(x.shape[0])
    distX[i, i] = 999
    div = np.min(distX, axis=1)
    div = -(div - np.min(div)) / (np.max(div) - np.min(div)) * (1-weight)

    if active:
        idx2 = np.argsort(div-ent)
    else:
        idx2 = np.argsort(ent+div)
    return idx1[idx2[:batch_size]]


def MMCS_MDS(x_p, y_p, batch_size, model, isdeep=True, active=False, weight=0.5):
    if isdeep:
        y_pred_pro = model.predict(x_p)
        x_feature_layer = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
        x_feature = x_feature_layer.predict(x_p)
    else:
        y_pred_pro = model.predict_proba(x_p)
        x_feature = x_p

    idx1 = np.argwhere(y_p != 1)
    idx1 = idx1.squeeze()
    MMC = y_pred_pro[idx1][:, 1] * np.linalg.norm(x_feature[idx1], axis=1)
    MMC = (MMC - np.min(MMC)) / (np.max(MMC) - np.min(MMC)) * weight

    x = x_feature[idx1]
    distX = squareform(pdist(x))
    i = np.arange(x.shape[0])
    distX[i, i] = 999
    div = np.min(distX, axis=1)
    div = -(div - np.min(div)) / (np.max(div) - np.min(div)) * (1-weight)

    if active:
        idx2 = np.argsort(div - MMC)
    else:
        idx2 = np.argsort(MMC + div)
    return idx1[idx2[:batch_size]]
