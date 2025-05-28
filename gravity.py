import cv2
import numpy as np
from vedo import *
from scipy.spatial import Delaunay
import triangle as tr
import time
import random
from vtkmodules.vtkRenderingCore import vtkRenderWindowInteractor
from vtkmodules.vtkRenderingCore import vtkCellPicker
from draw import auto_complete_contour, resample, get_constrained_triangulation, build_inflated_mesh
mesh_counter = 0

falling_meshes = []
falling_plotter = None

dragging = False
drag_target = None
last_mouse_pos = None

def random_color():
    return [random.random(), random.random(), random.random()]

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
    )

def fall_with_gravity(mesh, floor_z=0, drop_height=300, steps=60):
    import random
    global falling_meshes, falling_plotter

    if falling_plotter is None:
        init_falling_scene()

    # randomly generate a position
    rand_x = random.uniform(-500, 100)
    rand_y = random.uniform(-500, 100)
    rand_z = random.uniform(drop_height, drop_height + 100)

    mesh.pos(rand_x, rand_y, rand_z)
    falling_plotter += mesh
    falling_meshes.append(mesh)

    for i in range(steps):
        dz = (rand_z - floor_z) * (1 - (i + 1) / steps)
        mesh.pos(rand_x, rand_y, dz + floor_z)
        falling_plotter.render()
        time.sleep(0.01)

    falling_plotter.render()


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
                if picked == m.actor:  
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
                # adjust position
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
                print("Failed to make a mesh due to: ", e)

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
    