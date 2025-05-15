import cv2
import numpy as np
from vedo import Plotter, Mesh
from scipy.interpolate import splprep, splev
from scipy.spatial import Delaunay

# 初始化畫布
canvas_size = 512
img = np.zeros((canvas_size, canvas_size, 3), np.uint8)
drawing = False
points = []

def draw_callback(event, x, y, flags, param):
    global drawing, points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        points.append((x, y))
        if len(points) > 1:
            cv2.line(img, points[-2], points[-1], (255, 255, 255), 2)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

# 建立畫布視窗
cv2.namedWindow("Sketch")
cv2.setMouseCallback("Sketch", draw_callback)

# 建立 3D 場景
plt = Plotter(interactive=False, axes=1, size=(700,700))
z_offset = 0
models = []

def resample(contour, n=100):
    contour = np.array(contour)
    if np.linalg.norm(contour[0] - contour[-1]) > 5:
        contour = np.vstack([contour, contour[0]])
    k = min(3, len(contour)-1)
    tck, u = splprep([contour[:,0], contour[:,1]], s=0, per=True, k=k)
    unew = np.linspace(0, 1.0, n)
    out = splev(unew, tck)
    return np.vstack(out).T

def contour_to_3d(contour, z_shift=0):
    resampled = resample(contour, 100)
    tri = Delaunay(resampled)
    center = np.mean(resampled, axis=0)
    max_dist = np.max(np.linalg.norm(resampled - center, axis=1))
    z_top = [(1 - (np.linalg.norm(p - center) / max_dist)**2) * 20 for p in resampled]
    z_top = np.array(z_top)
    z_bot = -z_top

    top_pts = np.hstack([resampled, z_top[:, np.newaxis]])
    bot_pts = np.hstack([resampled, z_bot[:, np.newaxis]])
    top_pts[:,2] += z_shift
    bot_pts[:,2] += z_shift

    pts = np.vstack([top_pts, bot_pts])
    faces = []

    for tri_pts in tri.simplices:
        faces.append([tri_pts[0], tri_pts[1], tri_pts[2]])
    for tri_pts in tri.simplices:
        a, b, c = tri_pts + len(resampled)
        faces.append([c, b, a])

    n = len(resampled)
    for i in range(n):
        a, b = i, (i+1)%n
        a_bot, b_bot = a+n, b+n
        faces.append([a, b, b_bot])
        faces.append([a, b_bot, a_bot])

    mesh = Mesh([pts, faces])
    mesh.color("tomato").lighting("plastic").compute_normals()
    return mesh

# 主循環：畫筆 + Enter 輸出 → 疊加模型
while True:
    canvas_copy = img.copy()
    cv2.putText(canvas_copy, "畫圖後按 Enter 建模, 按 ESC 結束", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow("Sketch", canvas_copy)

    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter key
        if len(points) > 10:
            mesh = contour_to_3d(points, z_shift=z_offset)
            models.append(mesh)
            mesh.write(f"model_{len(models)}.obj")
            print(f"✅ 已新增模型並儲存為 model_{len(models)}.obj")
            z_offset -= 40  # 疊在下一層
            plt.clear()
            plt.show(models, resetcam=False, viewup="z")
            print("✅ 已新增模型！")
        else:
            print("⚠️ 點太少，請畫出明確輪廓")
        img[:] = 0
        points = []
    elif key == 27:  # ESC key
        break

cv2.destroyAllWindows()
plt.interactive().close()
