from dataclasses import dataclass
import numpy as np
import copy
import os

# =========================================================
# Mô hình dữ liệu
# =========================================================
@dataclass
class Individual:
    rnvec: np.ndarray          # vector quyết định, chuẩn hoá trong [0,1]
    factorial_costs: float = None
    constraint_violation: float = None

@dataclass
class AlgoParams:
    DE_F: float = 0.5          # hệ số đột biến (thường 0.3 ~ 0.9)
    DE_CR: float = 0.9         # xác suất crossover (thường 0.7 ~ 0.9)

# =========================================================
# KMeans (dùng sklearn nếu có, không thì fallback đơn giản)
# =========================================================
def _kmeans_numpy_fallback(X, n_clusters, n_init=5, max_iter=100, rng=None):
    """KMeans rất gọn bằng NumPy (fallback)."""
    rng = np.random.default_rng(rng)
    best_inertia = None
    best_labels = None
    for _ in range(n_init):
        # chọn ngẫu nhiên tâm ban đầu
        centroids = X[rng.choice(len(X), size=n_clusters, replace=False)].copy()
        for _ in range(max_iter):
            # gán nhãn gần nhất
            d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            labels = d2.argmin(axis=1)
            # cập nhật tâm
            new_centroids = np.vstack([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centroids[k]
                for k in range(n_clusters)
            ])
            if np.allclose(new_centroids, centroids):
                break
            centroids = new_centroids
        inertia = ((X - centroids[labels])**2).sum()
        if best_inertia is None or inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
    return best_labels

USE_SKLEARN_KMEANS = os.environ.get("RL_CMTEA_USE_SKLEARN", "").lower() in {"1", "true", "yes"}


def kmeans_labels(X, n_clusters, rng=None):
    if USE_SKLEARN_KMEANS:
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0)
            return km.fit_predict(X)
        except Exception:
            pass
    return _kmeans_numpy_fallback(X, n_clusters, rng=rng)

# =========================================================
# DE crossover (binomial)
# =========================================================
def de_crossover(v, u, CR, rng=None):
    """
    v: mutant vector
    u: target (current block)
    CR: crossover rate
    Đảm bảo ít nhất 1 vị trí lấy từ v (jrand).
    """
    rng = np.random.default_rng(rng)
    D = v.shape[-1]
    jrand = rng.integers(0, D)
    mask = rng.random(D) < CR
    mask[jrand] = True
    trial = np.where(mask, v, u)
    return trial

# =========================================================
# correDecode tương đương MATLAB
# =========================================================
def corre_decode(population, corre_row, divD):
    """
    population: list[list[Individual]]
    corre_row: [task_idx, ind_idx, start, end] (1-based trong MATLAB)
    Trả về block có kích thước đúng divD (có padding 0 nếu ngắn).
    """
    t, i, start, end = corre_row  # các chỉ số đang là 1-based từ MATLAB
    t_idx = t - 1
    i_idx = i - 1
    start_idx = start - 1
    end_idx = end  # python slice exclusive => dùng end trực tiếp

    seg = population[t_idx][i_idx].rnvec[start_idx:end_idx]
    if seg.size == divD:
        return seg.copy()
    # padding
    out = np.zeros(divD, dtype=float)
    out[:seg.size] = seg
    return out

# =========================================================
# Hàm KT chính
# =========================================================
def KT(algo: AlgoParams, tasks_dims, population, divK: int, divD: int, rng=None):
    """
    algo: AlgoParams(DE_F, DE_CR)
    tasks_dims: list[int] số chiều cho mỗi task
    population: list[list[Individual]], population[task][i]
    divK: số cụm kmeans
    divD: độ dài block
    Trả về: offspring (deepcopy của population, nhưng đã cập nhật các block)
    """
    rng = np.random.default_rng(rng)
    # -----------------------------------------------------
    # 1) Xây corre: mỗi dòng là [task, ind, start_dim, end_dim] (1-based)
    # -----------------------------------------------------
    max_dim = int(max(tasks_dims))
    corre = []
    for t in range(1, len(tasks_dims) + 1):  # 1-based
        n_ind = len(population[t-1])
        for i in range(1, n_ind + 1):        # 1-based
            n_blocks = int(np.ceil(max_dim / divD))
            for j in range(1, n_blocks + 1): # 1-based
                start = 1 + (j - 1) * divD
                end = min(j * divD, max_dim)
                corre.append([t, i, start, end])
    corre = np.array(corre, dtype=int)

    # -----------------------------------------------------
    # 2) Giải mã tất cả block thành ma trận dimVal (n_block x divD)
    # -----------------------------------------------------
    dimVal = np.vstack([corre_decode(population, row, divD) for row in corre])

    # -----------------------------------------------------
    # 3) KMeans chia thành divK cụm
    # -----------------------------------------------------
    idx = kmeans_labels(dimVal, divK, rng=rng)
    subpops = [[] for _ in range(divK)]
    for r, k in enumerate(idx):
        subpops[k].append(r)  # lưu chỉ số hàng trong corre/dimVal

    # -----------------------------------------------------
    # 4) Trong từng cụm: DE trên block, tạo offspring_temp + off_corre
    # -----------------------------------------------------
    offspring_temp = []
    off_corre = []

    for k in range(divK):
        members = subpops[k]  # các chỉ số hàng trong corre/dimVal
        m = len(members)
        if m < 4:
            continue
        # Duyệt từng phần tử trong cụm
        for pos_idx, row_idx in enumerate(members):
            # làm giống MATLAB: chọn 4 phần tử, bỏ phần tử đang xét, lấy 3 còn lại
            if m < 4:
                continue
            # random 4 distinct indices trong [0, m)
            A_local = rng.choice(m, size=4, replace=False)
            # map sang chỉ số toàn cục
            A_global = [members[a] for a in A_local]
            # nếu có trùng với row_idx thì bỏ
            A_global = [g for g in A_global if g != row_idx]
            if len(A_global) < 3:
                # fallback: chọn ngẫu nhiên 3 khác row_idx
                candidates = [g for g in members if g != row_idx]
                if len(candidates) < 3:
                    continue
                A_global = list(rng.choice(candidates, size=3, replace=False))

            r1, r2, r3 = A_global[:3]
            dp1 = dimVal[r1]
            dp2 = dimVal[r2]
            dp3 = dimVal[r3]

            # mutant v = dp1 + F * (dp2 - dp3)
            v = dp1 + algo.DE_F * (dp2 - dp3)
            v = np.clip(v, 0.0, 1.0)

            # target u = block của phần tử đang xét
            u = dimVal[row_idx]
            trial = de_crossover(v, u, algo.DE_CR, rng=rng)

            offspring_temp.append(trial)
            off_corre.append(corre[row_idx])

    offspring_temp = np.array(offspring_temp) if len(offspring_temp) else np.empty((0, divD))
    off_corre = np.array(off_corre, dtype=int) if len(off_corre) else np.empty((0, 4), dtype=int)

    # -----------------------------------------------------
    # 5) Ghi lại vào offspring (deep copy để không phá dữ liệu gốc)
    # -----------------------------------------------------
    offspring = copy.deepcopy(population)

    for i in range(off_corre.shape[0]):
        t, ind, start, end = off_corre[i]
        t_idx = t - 1
        ind_idx = ind - 1
        start_idx = start - 1
        end_idx = end
        # chỉ ghi phần thực (không ghi phần padding 0 thừa)
        seg_len = end - start + 1
        offspring[t_idx][ind_idx].rnvec[start_idx:end_idx] = offspring_temp[i][:seg_len]

        # đảm bảo [0,1]
        np.clip(offspring[t_idx][ind_idx].rnvec, 0.0, 1.0, out=offspring[t_idx][ind_idx].rnvec)

    return offspring

# =========================================================
# (Tuỳ chọn) Ví dụ nhỏ cách gọi
# =========================================================
if __name__ == "__main__":
    # Giả lập 2 task với số chiều khác nhau
    tasks_dims = [22, 30]
    pop_size = 5
    rng = np.random.default_rng(123)

    # Tạo population: 2 task, mỗi task 5 cá thể
    population = []
    for d in tasks_dims:
        task_pop = [Individual(rnvec=rng.random(d)) for _ in range(pop_size)]
        population.append(task_pop)

    algo = AlgoParams(DE_F=0.5, DE_CR=0.9)
    divK = 4
    divD = 10

    offspring = KT(algo, tasks_dims, population, divK=divK, divD=divD, rng=42)
    # offspring bây giờ là quần thể mới với các block đã chuyển giao tri thức
