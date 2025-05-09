from vedo import *
import numpy as np

mesh = Mesh("teddy_generated.obj")

cut_points = []

def on_click(evt):
    if not evt.actor:
        return
    pos = evt.picked3d
    cut_points.append(pos)
    print(f"點擊位置: {pos}")
    if len(cut_points) == 2:
        plt.close()

plt = Plotter(title="點擊兩下定義切割線", axes=1)
plt.add(mesh)
plt.add_callback("mouse click", on_click)
plt.show(interactive=True)

if len(cut_points) != 2:
    print("你必須點擊兩次以定義切割線！")
    exit()

p1, p2 = np.array(cut_points[0]), np.array(cut_points[1])
cut_dir = p2 - p1
cut_normal = np.cross(cut_dir, [0, 0, 1])
if np.linalg.norm(cut_normal) == 0:
    cut_normal = [1, 0, 0]
cut_normal = cut_normal / np.linalg.norm(cut_normal)

# 修正這裡：用 origin 和 normal 參數，而不是傳 vedo.Plane
cloned = mesh.clone()
cut_result = cloned.cut_with_plane(origin=p1, normal=cut_normal)

cut_result.color("lightblue").lighting("plastic")
cut_result.write("teddy_cut.obj")
print("✅ 已切割並匯出為 teddy_cut.obj")

# 顯示結果與平面（可選）
cut_plane = Plane(pos=p1, normal=cut_normal)
plt = Plotter(title="切割後模型", axes=1)
plt.show(cut_result, cut_plane, viewup="z")
