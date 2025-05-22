import cv2
import numpy as np
from vedo import *
from scipy.spatial import Delaunay
import triangle as tr
import time
import random
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker
mesh_counter = 0

falling_meshes = []
falling_plotter = None

dragging = False
drag_target = None
last_mouse_pos = None

def random_color():
    return [random.random(), random.random(), random.random()]

def build_inflated_mesh(verts2d, tris, contour, z_strength=30):
    center = np.mean(verts2d, axis=0)
    max_dist = np.max(np.linalg.norm(verts2d - center, axis=1))
    z_vals = []
    for pt in verts2d:
        d = np.linalg.norm(pt - center)
        z = np.cos(d / max_dist * np.pi / 2) * z_strength
        z_vals.append(z)

    verts3d = np.hstack([verts2d, np.array(z_vals)[:, None]])
    verts3d_bot = verts3d.copy()
    verts3d_bot[:, 2] *= -1
    verts3d_full = np.vstack([verts3d, verts3d_bot])

    top_indices = np.arange(len(verts2d))
    bot_indices = top_indices + len(verts2d)

    faces = []
    tris_bot = tris[:, ::-1] + len(verts3d)
    faces.extend(tris.tolist())
    faces.extend(tris_bot.tolist())

    segments = [[i, (i + 1) % len(verts2d)] for i in range(len(verts2d))]
    for i, j in segments:
        a, b = top_indices[i], top_indices[j]
        a2, b2 = bot_indices[i], bot_indices[j]
        faces.append([a, b, b2])
        faces.append([a, b2, a2])

    return Mesh([verts3d_full, faces]).compute_normals().lighting("plastic").color("orange")

def get_constrained_triangulation(contour):
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    vertices = contour.astype(np.float64)
    segments = [[i, (i + 1) % len(vertices)] for i in range(len(vertices))]
    A = dict(vertices=vertices, segments=segments)
    T = tr.triangulate(A, 'p')
    return T


def init_falling_scene():
    global falling_plotter
    falling_plotter = Plotter(title="Falling Meshes", axes=1, interactive=False)
    falling_plotter.picker = vtkCellPicker()
    floor = Plane(pos=(0, 0, 0), s=(600, 600)).color("gray5").alpha(0.3)
    falling_plotter += floor
    enable_drag_control(falling_plotter)
    falling_plotter.show(
        resetcam=True,
        interactive=False,
        size=(1200, 900),
        viewup="z"  # ✅ 關鍵設定：z 軸朝上
    )

def fall_with_gravity(mesh, floor_z=0, drop_height=300, steps=60):
    import random
    global falling_meshes, falling_plotter

    if falling_plotter is None:
        init_falling_scene()

    # 隨機位置與高度
    rand_x = random.uniform(-500, 100)
    rand_y = random.uniform(-500, 100)
    rand_z = random.uniform(drop_height, drop_height + 100)  # 更高一點

    mesh.pos(rand_x, rand_y, rand_z)
    falling_plotter += mesh
    falling_meshes.append(mesh)

    for i in range(steps):
        dz = (rand_z - floor_z) * (1 - (i + 1) / steps)
        mesh.pos(rand_x, rand_y, dz + floor_z)
        falling_plotter.render()
        time.sleep(0.01)

    falling_plotter.render()
def screen_to_world_on_plane(plotter, x, y, target_z):
    cam = plotter.camera
    renderer = plotter.renderer
    w2i = renderer.GetActiveCamera().GetCompositeProjectionTransformMatrix(renderer.GetTiledAspectRatio(), -1, 1)
    renderer.SetDisplayPoint(x, y, 0)
    renderer.DisplayToWorld()
    world_point1 = np.array(renderer.GetWorldPoint()[:3])

    renderer.SetDisplayPoint(x, y, 1)
    renderer.DisplayToWorld()
    world_point2 = np.array(renderer.GetWorldPoint()[:3])

    # 射線兩點 world_point1 (near), world_point2 (far)
    direction = world_point2 - world_point1
    if direction[2] == 0:
        return world_point1  # 平行於平面，避免除以 0

    t = (target_z - world_point1[2]) / direction[2]
    intersection = world_point1 + t * direction
    return intersection

def enable_drag_control(plotter):
    global dragging, drag_target, drag_offset

    if not hasattr(plotter, "picker") or plotter.picker is None:
        from vtkmodules.vtkRenderingCore import vtkCellPicker
        plotter.picker = vtkCellPicker()

    def left_button_press(obj, event):
        global dragging, drag_target, drag_offset
        ctrl_pressed = plotter.interactor.GetControlKey()
        click_pos = plotter.interactor.GetEventPosition()
        if ctrl_pressed:
            picker = plotter.picker
            picker.Pick(click_pos[0], click_pos[1], 0, plotter.renderer)
            picked = picker.GetActor()
            for m in falling_meshes:
                if picked == m.actor:  # ← 這裡改成 m.actor
                    dragging = True
                    drag_target = m
                    pick_pos = np.array(picker.GetPickPosition())
                    mesh_center = np.array(m.center_of_mass())
                    drag_offset = mesh_center - pick_pos
                    return
        dragging = False
        drag_target = None
        drag_offset = None

    def mouse_move(obj, event):
        global dragging, drag_target
        if dragging and drag_target is not None:
            mouse_pos = plotter.interactor.GetEventPosition()
            picker = plotter.picker
            picker.Pick(mouse_pos[0], mouse_pos[1], 0, plotter.renderer)
            pick_pos = np.array(picker.GetPickPosition())
            if np.linalg.norm(pick_pos) > 1e-6:
                current_z = drag_target.center_of_mass()[2]
                pick_pos[2] = current_z
                # ✅ 修正偏移 (200, 200)
                pick_pos[0] -= 250
                pick_pos[1] -= 250
                drag_target.pos(pick_pos)
                plotter.render()

    def left_button_release(obj, event):
        global dragging, drag_target, drag_offset
        dragging = False
        drag_target = None
        drag_offset = None

    iren = plotter.interactor
    iren.AddObserver("LeftButtonPressEvent", left_button_press)
    iren.AddObserver("MouseMoveEvent", mouse_move)
    iren.AddObserver("LeftButtonReleaseEvent", left_button_release)
    
def drawing2():
    global mesh_counter
    global falling_plotter
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

    img = np.zeros((512, 512, 3), np.uint8)
    cv2.namedWindow("Draw Contour")
    cv2.setMouseCallback("Draw Contour", draw)

    while True:
        cv2.imshow("Draw Contour", img)
        key = cv2.waitKey(1) & 0xFF

        if key == 13 and points:  # Enter
            contour = np.array(points, dtype=np.int32)

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

            def resample(contour, n=100):
                from scipy.interpolate import splprep, splev
                if len(contour) < 4:
                    contour = np.repeat(contour[:1], 4, axis=0)
                if np.linalg.norm(contour[0] - contour[-1]) > 5:
                    contour = np.vstack([contour, contour[0]])
                if np.max(np.linalg.norm(contour - contour[0], axis=1)) < 5:
                    contour = contour + np.random.normal(0, 1, contour.shape)
                _, unique_idx = np.unique(contour, axis=0, return_index=True)
                contour = contour[np.sort(unique_idx)]
                try:
                    k = min(3, len(contour) - 1)
                    tck, u = splprep([contour[:, 0], contour[:, 1]], s=0, per=True, k=k)
                    unew = np.linspace(0, 1.0, n)
                    out = splev(unew, tck)
                    return np.vstack(out).T
                except:
                    return np.linspace(contour[0], contour[-1], n)

            contour = auto_complete_contour(contour)
            if np.linalg.norm(contour[0] - contour[-1]) > 5:
                contour = np.vstack([contour, contour[0]])
            try:
                resampled = resample(contour, 100)
                tri_data = get_constrained_triangulation(resampled)
                verts2d = tri_data["vertices"]
                tris = tri_data["triangles"]
                mesh = build_inflated_mesh(verts2d, tris, resampled, z_strength=80)
                mesh.c(random_color())
                mesh_counter += 1
                mesh.write(f"savedObjects/teddy_generated_{mesh_counter}.obj")
                rand_x = random.uniform(-1000, 1000)
                rand_y = random.uniform(-1000, 1000)
                rand_z = random.uniform(250, 400)
                mesh.pos(rand_x, rand_y, rand_z)
                fall_with_gravity(mesh, drop_height=rand_z)
            except Exception as e:
                print("⚠️ Mesh 建立失敗：", e)

            img[:] = 0
            points.clear()

        elif key == ord('c'):  # clear all mesh
            if falling_plotter:
                for m in falling_meshes:
                    falling_plotter.remove(m)
                falling_meshes.clear()
                falling_plotter.render()

        elif key == 27:  # ESC
            cv2.destroyWindow("Draw Contour")
            if falling_plotter:
                falling_plotter.close()
                falling_plotter = None
            break
    