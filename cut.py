from vedo import *
import numpy as np
import time
from vtkmodules.vtkFiltersCore import vtkCutter

def draw_and_extrude_curve(mesh):
    """
    Allow user to draw curves on the surface and extrude them along camera view direction
    """
    plt = Plotter(axes=1, bg='white', bg2='lightgray')
    
    # Set mesh properties for better visibility
    # mesh.color('lightblue').alpha(0.9).lighting('glossy')
    # mesh.wireframe(True).lineWidth(0.5)
    
    # Drawing state variables
    # Drawing state variables
    is_drawing = False
    points = []
    curve = None
    markers = []
    cut_mesh = None
    extruded_surface = None
    previous_mesh = None  # Store the previous state of the mesh
    extrusion_path = None
    # Undo/redo stacks
    deleted_points = []
    deleted_markers = []
    def on_left_click(evt):
        nonlocal points, curve, markers
        
        if not is_drawing or not evt.actor or evt.actor != mesh:
            return
            
        # Get clicked point on the mesh surface
        pt = mesh.closest_point(evt.picked3d)
        points.append(pt)
        
        # Add a small sphere marker at clicked point
        marker = Sphere(pt, r=0.02, c='green')
        markers.append(marker)
        plt.add(marker)
        
        # Update the curve if we have at least 2 points
        if len(points) > 1:
            if curve:
                plt.remove(curve)
            curve = Line(points, lw=4, c='red')
            plt.add(curve)
        
        plt.render()
    
    def extrude_curve_along_view():
        nonlocal points, extruded_surface, extrusion_path
        
        if len(points) < 2:
            print("Need at least 2 points to create a surface")
            return None
        
        # Get current camera parameters
        cam = plt.camera
        view_direction = np.array(cam.GetDirectionOfProjection())
        view_direction /= np.linalg.norm(view_direction)  # Normalize
        
        # Create extrusion path (opposite of view direction)
        extrusion_length = mesh.diagonal_size() * 0.5  # Use half of mesh size
        path_start = np.array(points[-1])  # Start from last point
        path_end = path_start + view_direction * extrusion_length
        
        # Create a line to represent the extrusion path
        extrusion_path = Line(path_start, path_end, c='blue', lw=2, alpha=0.5)
        
        # Create ruled surface between curve and extruded curve
        extruded_points = [pt + view_direction * extrusion_length for pt in points]
        
        # Combine original and extruded points
        all_points = points + extruded_points
        
        # Create triangles for the surface
        triangles = []
        n = len(points)
        for i in range(n-1):
            # First triangle
            triangles.append([i, i+1, n+i])
            # Second triangle
            triangles.append([i+1, n+i+1, n+i])
        
        # Create the extruded surface
        extruded_surface = Mesh([all_points, triangles])
        extruded_surface.color('orange').alpha(0.6).lighting('glossy')
        
        return extrusion_path, extruded_surface
    
    def perform_precise_cut():
        nonlocal cut_mesh, extruded_surface, mesh, previous_mesh, extrusion_path
        
        if not extruded_surface:
            print("No extruded surface to cut with")
            return None
        
        try:
            previous_mesh = mesh.clone()
            # Option 1: Boolean cut (more accurate but may fail on complex cases)
            cut_mesh = mesh.clone().cut_with_mesh(extruded_surface)
            
            cut_mesh = cut_mesh.cap()
            # Option 2: Slice the mesh (alternative approach)
            # cut_mesh = mesh.clone().slice_with_surface(extruded_surface)
            
            # if cut_mesh.N() == 0:
            #     print("Cut operation produced empty result")
            #     return None
            
            # Color the cut mesh and show the cut edge
            cut_mesh.c('purple').alpha(1.0).lighting('glossy')
            # cut_mesh.lineWidth(1.5).lineColor('red')

            plt.remove(extruded_surface)
            plt.remove(extrusion_path)
            extruded_surface = None
            
            return cut_mesh
        
        except Exception as e:
            print(f"Error during cutting: {e}")
            return None

    def on_key_press(evt):
        nonlocal is_drawing, points, curve, markers, extruded_surface, cut_mesh, mesh, previous_mesh
        
        if evt.keypress == 'space':
            # Toggle drawing state
            is_drawing = not is_drawing
            
            if is_drawing:
                print("Drawing STARTED - click on the surface to add points")
                # Clear previous points if starting new drawing
                points = []
                if curve:
                    plt.remove(curve)
                    curve = None
                for m in markers:
                    plt.remove(m)
                markers = []
                if extruded_surface:
                    plt.remove(extruded_surface)
                    extruded_surface = None
            else:
                print("Drawing STOPPED")
                if len(points) > 1:
                    print(f"Recorded {len(points)} points")
                
        elif evt.keypress == 'c':
            # Clear current curve
            if curve:
                plt.remove(curve)
                curve = None
            points = []
            for m in markers:
                plt.remove(m)
            markers = []
            if extruded_surface:
                plt.remove(extruded_surface)
                extruded_surface = None
            plt.render()
            print("Current curve cleared")
            
        elif evt.keypress == 's':
            # Save the cut mesh
            if cut_mesh is not None:
                try:
                    filename = "cut_model.obj"
                    cut_mesh.write(filename)
                    print(f"Cut mesh saved to {filename}")
                except Exception as e:
                    print(f"Error saving mesh: {e}")
            else:
                print("No cut mesh available to save - perform a cut first")
        
        elif evt.keypress == 'e':
            # Extrude the curve along view direction
            if len(points) > 1:
                if extruded_surface:
                    plt.remove(extruded_surface)
                
                extrusion_path, extruded_surface = extrude_curve_along_view()
                plt.add(extrusion_path)
                plt.add(extruded_surface)
                print("Created extruded surface along view direction")
            else:
                print("No valid curve to extrude")
        
        elif evt.keypress == '1':
            mesh.color('lightblue')
            plt.render()
        
        elif evt.keypress == '2':
            mesh.color('lightgreen')
            plt.render()
        
        elif evt.keypress == '3':
            mesh.color('pink')
            plt.render()
        elif evt.keypress == 'x':
            if extruded_surface:
                cut_mesh = perform_precise_cut()
                if cut_mesh:
                    plt.add(cut_mesh)
                    print("Precise cut performed")
                    plt.render()
            else:
                print("No extruded surface - create one with 'e' first")
        elif evt.keypress == 'u':
            # Undo the cut operation
            if previous_mesh is not None:
                plt.remove(cut_mesh)  # Remove the cut mesh
                mesh = previous_mesh  # Restore the previous mesh state
                plt.add(mesh)        # Add the restored mesh back to the scene
                previous_mesh = None  # Clear the previous mesh reference
                cut_mesh = None      # Clear the cut mesh reference
                plt.render()
                print("Cut operation undone")
            else:
                print("Nothing to undo")
    
    plt.add_callback('LeftButtonPress', on_left_click)
    plt.add_callback('KeyPress', on_key_press)
    
    # Instructions
    print("\nINSTRUCTIONS:")
    print("- Press SPACEBAR to start/stop drawing")
    print("  - Left-click on the mesh to add points")
    print("- Press 'e' to extrude curve along view direction")
    print("- Press 'x' to cut the mesh with the extruded surface")
    print("- Press 'c' to clear current curve and surface")
    print("- Press 's' to save the cutted object")
    print("- Press 'u' to undo last point")
    print("- Press '1'/'2'/'3' to change mesh color")
    print("- Press 'q' to quit")
    
    # Add status indicator light
    # light = Light(pos=[0,0,0], c='gray', radius=0.1)
    # plt.add(light)
    
    # def update_light():
    #     light.color('green' if is_drawing else 'red')
    #     plt.render()
    # 
    # # Show the plot
    plt.show(mesh, "3D Surface Drawing Tool").parallel_projection(False)
    # 
    # # Keep the light updated
    # while plt.escaped is False:
    #     update_light()
    #     plt.processEvents()
    #     time.sleep(0.05)

def main():
    # Load a 3D object (replace with your file path)
    # file_path = "path/to/your/model.obj"  # Supports .obj, .stl, .ply, etc.
    
    try:
        # Load the mesh with initial color
        mesh = Mesh("teddy_generated.obj")
        
        # Set initial mesh properties
        # mesh.color('lightblue').alpha(0.9).lighting('glossy')
        
        # print(f"Mesh loaded: {file_path}")
        # print(f"Number of vertices: {mesh.N()}")
        # print(f"Number of faces: {mesh.NCells()}")
        
        draw_and_extrude_curve(mesh)
        
    except Exception as e:
        print(f"Error loading 3D object: {e}")

if __name__ == "__main__":
    main()
