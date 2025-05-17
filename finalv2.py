import cv2
import numpy as np
from vedo import *
from scipy.spatial import Delaunay, distance_matrix
import matplotlib.pyplot as plt
import triangle as tr

def get_constrained_triangulation(contour):
    """
    Perform constrained Delaunay triangulation with triangle lib.
    """
    # Remove duplicate point at the end
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]

    # Convert to triangle format
    vertices = contour.astype(np.float64)
    segments = [[i, (i + 1) % len(vertices)] for i in range(len(vertices))]

    A = dict(vertices=vertices, segments=segments)
    T = tr.triangulate(A, 'p')  # 'p': use PSLG input (preserve segments)

    return T

def classify_triangles(vertices, triangles, segments):
    """
    Classify triangles into terminal / sleeve / junction
    Return:
        - internal_edges: list of edges not in boundary
        - triangle_types: dict triangle index -> type
    """
    # 將所有 segments 轉成無序邊集合
    boundary_edges = set(tuple(sorted(e)) for e in segments)
    triangle_types = {}
    internal_edges = set()

    for i, tri in enumerate(triangles):
        edges = [tuple(sorted([tri[j], tri[(j+1)%3]])) for j in range(3)]
        ext_cnt = sum(e in boundary_edges for e in edges)
        if ext_cnt == 2:
            triangle_types[i] = "terminal"
        elif ext_cnt == 1:
            triangle_types[i] = "sleeve"
        else:
            triangle_types[i] = "junction"

        # 非邊界的 edge 加入 internal_edges
        for e in edges:
            if e not in boundary_edges:
                internal_edges.add(e)

    return triangle_types, list(internal_edges)

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

    # Constrained Delaunay triangulation using triangle
    tri_data = get_constrained_triangulation(resampled)
    verts2d = tri_data["vertices"]
    tris = tri_data["triangles"]

    # 分類三角形與找 internal edges
    segments = [[i, (i + 1) % len(verts2d)] for i in range(len(verts2d))]
    triangle_types, internal_edges = classify_triangles(verts2d, tris, segments)

    # 建立 chordal axis（內部邊中點）
    spine_pts = []
    for e in internal_edges:
        p1, p2 = verts2d[e[0]], verts2d[e[1]]
        mid = (p1 + p2) / 2
        spine_pts.append(mid)
    spine_pts = np.array(spine_pts)

    # 根據 spine elevation 計算 z 值（與相鄰 boundary vertex 距離平均）
    z_vals = []
    for pt in spine_pts:
        dists = np.linalg.norm(verts2d - pt, axis=1)
        avg_dist = np.mean(np.partition(dists, 5)[:5])  # 最近 5 個
        z = avg_dist * 0.5  # 膨脹係數可調整
        z_vals.append(z)
    z_vals = np.array(z_vals)

    # 建立上半部 3D mesh 頂點（spine + 外輪廓）
    spine3d = np.hstack([spine_pts, z_vals[:, None]])
    boundary3d = np.hstack([verts2d, np.zeros((len(verts2d), 1))])
    points3d = np.vstack([boundary3d, spine3d])

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

    # for tri_pts in tri.simplices:
    #     faces.append([tri_pts[0], tri_pts[1], tri_pts[2]])  # 上面
    # for tri_pts in tri.simplices:
    #     a, b, c = tri_pts + len(resampled)
    #     faces.append([c, b, a])  # 底面反轉

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
