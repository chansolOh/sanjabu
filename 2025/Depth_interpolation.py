import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

from scipy.signal import convolve2d


def fill_depth_nearest(depth):
    depth = depth.astype(float)
    mask = depth > 0  # 유효 픽셀

    # 각 픽셀이 "가까운 유효 픽셀"의 인덱스를 가리키도록 함
    indices = distance_transform_edt(~mask,
                                     return_distances=False,
                                     return_indices=True)

    filled = depth[tuple(indices)]  # 최근접 유효 값으로 모두 채움
    return filled


def fast_depth_interpolate(depth, radius=3):
    depth = depth.astype(float)

    # 7×7 가중치 커널 생성 (거리 기반 인버스 weight)
    y, x = np.mgrid[-radius:radius+1, -radius:radius+1]
    dist = np.sqrt(x*x + y*y)
    dist[radius, radius] = 1  # 중심 0 → 1로 변경
    weight = 1.0 / dist
    weight /= weight.sum()

    # mask: 유효 depth 여부
    mask = (depth > 0).astype(float)

    # 가중 depth 합
    depth_conv = convolve2d(depth * mask, weight, mode='same', boundary='symm')

    # 가중치 합 (유효 depth만 반영)
    weight_conv = convolve2d(mask, weight, mode='same', boundary='symm')

    # 결과 보간
    out = depth.copy()
    missing = (depth == 0)
    out[missing] = depth_conv[missing] / (weight_conv[missing] + 1e-6)

    return out

# def interpolate_depth(depth, radius=3):
#     """
#     depth: H×W depth map (0 = invalid)
#     radius: 주변 픽셀 탐색 반경 (기본 3)
#     """
#     depth = depth.astype(float)
#     h, w = depth.shape

#     # invalid mask
#     invalid = (depth == 0)

#     # 거리 기반 weight 생성
#     dist = distance_transform_edt(invalid)
#     dist[dist == 0] = 1  # divide by zero 방지

#     # 결과 depth 복사
#     depth_interp = depth.copy()

#     # 보간할 좌표
#     ys, xs = np.where(invalid)

#     for y, x in zip(ys, xs):
#         y1, y2 = max(0, y - radius), min(h, y + radius + 1)
#         x1, x2 = max(0, x - radius), min(w, x + radius + 1)

#         patch = depth[y1:y2, x1:x2]
#         mask = patch > 0

#         if np.any(mask):
#             # 가까운 픽셀에 더 큰 weight 부여 (선형 가중)
#             dy, dx = np.indices(patch.shape)
#             dy = dy - (y - y1)
#             dx = dx - (x - x1)
#             dist2 = np.sqrt(dx*dx + dy*dy)
#             dist2[dist2 == 0] = 1
#             weights = (1.0 / dist2) * mask

#             depth_interp[y, x] = np.sum(patch * weights) / np.sum(weights)

#     return depth_interp

root_path = "/nas/Dataset/Dataset_2025/dataset_v1_real/Logistic_Site/UONRobitcs_1F/demo1/depth/top_view_camera"

if __name__ == "__main__":
    img = np.load(f"{root_path}/0003.npy")
    # interp_img = fast_depth_interpolate(img, radius=5)

    filled = fill_depth_nearest(img)
    filled[img == 0] = filled[img == 0] 
    fig,ax = plt.subplots(1,2, figsize=(12,6))
    im_0 = ax[0].imshow(img, cmap='turbo' )
    im_1 = ax[1].imshow(filled, cmap='turbo')

    plt.show()
