import cv2
import numpy as np
from vedo import *
from scipy.spatial import Delaunay
import triangle as tr

def get_constrained_triangulation(contour):
    if np.allclose(contour[0], contour[-1]):
        contour = contour[:-1]
    vertices = contour.astype(np.float64)
    segments = [[i, (i + 1) % len(vertices)] for i in range(len(vertices))]
    A = dict(vertices=vertices, segments=segments)
    T = tr.triangulate(A, 'p')
    return T

def classify_triangles(vertices, triangles, segments):
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
        for e in edges:
            if e not in boundary_edges:
                internal_edges.add(e)

    return triangle_types, list(internal_edges)

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

def drawing(name):
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
        if key == 13:
            break

    cv2.destroyAllWindows()

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
            raise ValueError("No lines are drawn.")
        return contour

    contour = auto_complete_contour(contour)

    if np.linalg.norm(contour[0] - contour[-1]) > 5:
        contour = np.vstack([contour, contour[0]])

    def resample(contour, n=100):
        from scipy.interpolate import splprep, splev

        # add points to 4 if less than 4
        if len(contour) < 4:
            contour = np.repeat(contour[:1], 4, axis=0)

        # make the skech closed
        if np.linalg.norm(contour[0] - contour[-1]) > 5:
            contour = np.vstack([contour, contour[0]])

        # add noise to the contour
        if np.max(np.linalg.norm(contour - contour[0], axis=1)) < 5:
            contour = contour + np.random.normal(0, 1, contour.shape)

        # remove duplicate points
        _, unique_idx = np.unique(contour, axis=0, return_index=True)
        contour = contour[np.sort(unique_idx)]

        try:
            k = min(3, len(contour) - 1)
            tck, u = splprep([contour[:, 0], contour[:, 1]], s=0, per=True, k=k)
            unew = np.linspace(0, 1.0, n)
            out = splev(unew, tck)
            return np.vstack(out).T

        except Exception as e:
            print("Failed to resample: ", e)

            if len(contour) < 2:
                contour = np.vstack([contour[0], contour[0] + [1, 1]])
            return np.linspace(contour[0], contour[-1], n)

    try:
        resampled = resample(contour, 100)
    except Exception as e:
        print("Failed to resample: ", e)
        return

    tri_data = get_constrained_triangulation(resampled)
    verts2d = tri_data["vertices"]
    tris = tri_data["triangles"]
    # z_strength: z
    mesh = build_inflated_mesh(verts2d, tris, resampled, z_strength=80)
    mesh.write("savedObjects/" + name + ".obj")
    plt = Plotter(title="Teddy Inflated Model", axes=1)
    plt.show(mesh, viewup="z", interactive=True)
    plt.close()  
