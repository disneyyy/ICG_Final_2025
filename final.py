# # from skimage.morphology import medial_axis
# # skeleton, _ = medial_axis(binary_image, return_distance=True)
# import cv2
# import numpy as np
# from vedo import *
# from scipy.spatial import Delaunay, distance_matrix
# import matplotlib.pyplot as plt

# def drawing():
#     drawing_flag = False
#     points = []

#     def draw(event, x, y, flags, param):
#         nonlocal drawing_flag, points  # 這裡用 nonlocal
#         if event == cv2.EVENT_LBUTTONDOWN:
#             drawing_flag = True
#             points = [(x, y)]
#         elif event == cv2.EVENT_MOUSEMOVE and drawing_flag:
#             points.append((x, y))
#             cv2.line(img, points[-2], points[-1], (255, 255, 255), 2)
#         elif event == cv2.EVENT_LBUTTONUP:
#             drawing_flag = False
#             points.append((x, y))
#             cv2.line(img, points[-2], points[-1], (255, 255, 255), 2)

#     # 初始化畫布
#     img = np.zeros((512, 512, 3), np.uint8)
#     cv2.namedWindow("Draw Contour")
#     cv2.setMouseCallback("Draw Contour", draw)

#     while True:
#         cv2.imshow("Draw Contour", img)
#         key = cv2.waitKey(1) & 0xFF
#         if key == 13:  # 按 enter 結束
#             break

#     cv2.destroyAllWindows()

#     contour = np.array(points, dtype=np.int32)
#     np.save("contour.npy", contour)


#     # 載入輪廓與重取樣
#     contour = np.load("contour.npy")
#     if np.linalg.norm(contour[0] - contour[-1]) > 5:
#         contour = np.vstack([contour, contour[0]])

#     def resample(contour, n=100):
#         from scipy.interpolate import splprep, splev
#         if len(contour) < 4:
#             raise ValueError(f"需要至少4個點才能進行樣條重取樣，目前只有 {len(contour)} 點")
#         k = min(3, len(contour)-1)
#         tck, u = splprep([contour[:,0], contour[:,1]], s=0, per=True, k=k)
#         unew = np.linspace(0, 1.0, n)
#         out = splev(unew, tck)
#         return np.vstack(out).T

#     resampled = resample(contour, 100)

#     # Delaunay triangulation
#     tri = Delaunay(resampled)

#     # Elevation: 對每個點計算到輪廓的最短距離，當作「膨脹高度」
#     D = distance_matrix(resampled, resampled)
#     border_idx = np.arange(len(resampled))  # 全部點都當外輪廓
#     max_dist = D.max()
#     # 計算中心點
#     center = np.mean(resampled, axis=0)

#     # 每個點與中心的距離
#     max_dist = np.max(np.linalg.norm(resampled - center, axis=1))

#     z_top = []
#     z_strength = 10  # 膨脹強度
#     for p in resampled:
#         d = np.linalg.norm(p - center)
#         # z = (1 - (d / max_dist)**2) * 20  # 二次拋物線曲面
#         z = np.cos(d / max_dist * np.pi / 2) * z_strength
#         z_top.append(z)
#     # z_top = []

#     # for i in range(len(resampled)):
#     #     d = np.min([np.linalg.norm(resampled[i] - resampled[j]) for j in border_idx])
#     #     z = (d / max_dist)**0.5 * 15  # 半圓形上凸，數值越大越高
#     #     z_top.append(z)

#     z_top = np.array(z_top)
#     z_bot = -z_top  # 對稱底部

#     # 構建點
#     top_pts = np.hstack([resampled, z_top[:, np.newaxis]])
#     bot_pts = np.hstack([resampled, z_bot[:, np.newaxis]])
#     points = np.vstack([top_pts, bot_pts])

#     # 構建面（頂部三角形 + 底部翻面三角形 + 側邊封起來）
#     faces = []

#     # 頂部面
#     for tri_pts in tri.simplices:
#         faces.append([tri_pts[0], tri_pts[1], tri_pts[2]])

#     # 底部面（翻轉順序）
#     for tri_pts in tri.simplices:
#         a, b, c = tri_pts + len(resampled)
#         faces.append([c, b, a])  # 反轉面向

#     # 側邊封口（環狀連接上下兩層）
#     n = len(resampled)
#     for i in range(n):
#         a, b = i, (i+1)%n
#         a_bot, b_bot = a+n, b+n
#         faces.append([a, b, b_bot])
#         faces.append([a, b_bot, a_bot])

#     # 建 mesh 並顯示
#     mesh = Mesh([points, faces])
#     mesh.color("orange").lighting("plastic").compute_normals()
#     mesh.write("teddy_generated.obj")

#     show(mesh, axes=1, title="Teddy-like 3D Model", viewup="z")
import cv2
import numpy as np
from vedo import *
from scipy.spatial import Delaunay, distance_matrix
import matplotlib.pyplot as plt

def drawing():
    drawing_flag = False
    points = []

    def draw(event, x, y, flags, param):
        nonlocal drawing_flag, points
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing_flag = True
            points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing_flag:
            points.append((x, y))
            cv2.line(img, points[-2], points[-1], (255, 255, 255), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing_flag = False
            points.append((x, y))
            cv2.line(img, points[-2], points[-1], (255, 255, 255), 2)

    # 初始化畫布
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow("Draw Contour")
    cv2.setMouseCallback("Draw Contour", draw)

    while True:
        cv2.imshow("Draw Contour", img)
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter
            break

    cv2.destroyAllWindows()

    # 將點轉成 numpy 陣列
    contour = np.array(points, dtype=np.int32)

    # 自動補齊至少 4 個點
    def auto_complete_contour(contour):
        if len(contour) >= 4:
            return contour
        elif len(contour) == 3:
            p = ((contour[0] + contour[1]) // 2).astype(int)
            contour = np.insert(contour, 1, p, axis=0)
        elif len(contour) == 2:
            p1 = ((contour[0] * 2 + contour[1]) // 3).astype(int)
            p2 = ((contour[0] + contour[1] * 2) // 3).astype(int)
            contour = np.vstack([contour[0], p1, p2, contour[1]])
        elif len(contour) == 1:
            contour = np.repeat(contour, 4, axis=0)
        else:
            raise ValueError("請至少畫一筆線段")
        return contour

    contour = auto_complete_contour(contour)

    # 確保首尾接起來
    if np.linalg.norm(contour[0] - contour[-1]) > 5:
        contour = np.vstack([contour, contour[0]])

    # 重取樣函數
    def resample(contour, n=100):
        from scipy.interpolate import splprep, splev
        k = min(3, len(contour) - 1)
        tck, u = splprep([contour[:, 0], contour[:, 1]], s=0, per=True, k=k)
        unew = np.linspace(0, 1.0, n)
        out = splev(unew, tck)
        return np.vstack(out).T

    try:
        resampled = resample(contour, 100)
    except Exception as e:
        print("樣條重取樣失敗：", e)
        return

    # Delaunay triangulation
    tri = Delaunay(resampled)

    # 中心點與膨脹高度計算
    center = np.mean(resampled, axis=0)
    max_dist = np.max(np.linalg.norm(resampled - center, axis=1))

    z_top = []
    z_strength = 10
    for p in resampled:
        d = np.linalg.norm(p - center)
        z = np.cos(d / max_dist * np.pi / 2) * z_strength
        z_top.append(z)

    z_top = np.array(z_top)
    z_bot = -z_top

    # 建立點陣列
    top_pts = np.hstack([resampled, z_top[:, np.newaxis]])
    bot_pts = np.hstack([resampled, z_bot[:, np.newaxis]])
    points3d = np.vstack([top_pts, bot_pts])

    # 建立面
    faces = []

    for tri_pts in tri.simplices:
        faces.append([tri_pts[0], tri_pts[1], tri_pts[2]])  # 上面
    for tri_pts in tri.simplices:
        a, b, c = tri_pts + len(resampled)
        faces.append([c, b, a])  # 底面反轉

    n = len(resampled)
    for i in range(n):
        a, b = i, (i + 1) % n
        a_bot, b_bot = a + n, b + n
        faces.append([a, b, b_bot])
        faces.append([a, b_bot, a_bot])

    # 顯示
    mesh = Mesh([points3d, faces])
    mesh.color("orange").lighting("plastic").compute_normals()
    mesh.write("teddy_generated.obj")
    show(mesh, axes=1, title="Teddy-like 3D Model", viewup="z")
