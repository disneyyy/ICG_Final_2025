import cv2
import numpy as np
from vedo import *
from scipy.spatial import Delaunay, distance_matrix
import matplotlib.pyplot as plt
import triangle as tr

def quarter_oval_sweep(p1, p2, height, steps=6):
    arc_pts = []
    p1, p2 = np.array(p1), np.array(p2)
    for t in np.linspace(0, np.pi / 2, steps):
        base = (1 - np.cos(t)) * p1 + np.cos(t) * p2
        z_offset = np.sin(t) * height
        arc_pts.append(np.append(base, z_offset))
    return np.array(arc_pts)

def build_oval_sweep_mesh(chord_edges, elevation_map):
    all_pts = []
    faces = []
    top_loops = []
    pt_offset = 0

    for (p1, p2) in chord_edges:
        mid = tuple(((np.array(p1) + np.array(p2)) / 2).tolist())
        height = elevation_map.get(mid, 1.0)
        arc = quarter_oval_sweep(p1, p2, height)
        arc_idx = list(range(pt_offset, pt_offset + len(arc)))
        all_pts.extend(arc)
        top_loops.append(arc_idx)
        pt_offset += len(arc)

    # 下半部鏡射，並對應 index
    bottom_loops = []
    for arc_idx in top_loops:
        start = len(all_pts)
        arc = [all_pts[i].copy() for i in arc_idx]
        for pt in arc:
            pt[2] *= -1
        all_pts.extend(arc)
        bottom_loops.append(list(range(start, start + len(arc))))

    # 建立側面三角形（縫合 top-bottom）
    for top, bot in zip(top_loops, bottom_loops):
        for i in range(len(top) - 1):
            t1, t2 = top[i], top[i + 1]
            b1, b2 = bot[i], bot[i + 1]
            faces.append([t1, t2, b2])
            faces.append([t1, b2, b1])

    # 建立上下蓋面（扇形）
    for loop in top_loops + bottom_loops:
        for i in range(len(loop) - 2):
            faces.append([loop[0], loop[i + 1], loop[i + 2]])

    # return Mesh([all_pts, faces]).compute_normals().lighting(\"plastic\").color(\"orange\")
    return Mesh([all_pts, faces]).compute_normals().lighting("plastic").color("orange")

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

    # 建立 chord_edges
    chord_edges = [(verts2d[e[0]], verts2d[e[1]]) for e in internal_edges]

    # 建立 elevation_map（中點作為 key，距外點平均距離作為 z 高度）
    elevation_map = {}
    for p1, p2 in chord_edges:
        mid = ((np.array(p1) + np.array(p2)) / 2)
        dists = np.linalg.norm(verts2d - mid, axis=1)
        avg_dist = np.mean(np.partition(dists, 5)[:5])
        elevation_map[tuple(mid)] = avg_dist * 0.5  # 膨脹係數可調整


    # 顯示
    mesh = build_oval_sweep_mesh(chord_edges, elevation_map)
    mesh.color("orange").lighting("plastic").compute_normals()
    mesh.write("teddy_generated.obj")
    show(mesh, axes=1, title="Teddy-like 3D Model", viewup="z")
