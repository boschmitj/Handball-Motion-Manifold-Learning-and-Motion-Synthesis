import numpy as np
import scenepic as sp


def build_scene(meshes, images=None, images_names=None, file_name = "mesh_animation"):
    ''' Create a scene with a mesh animation
    meshes: list of dictionaries with the following
        - name: name of the mesh
        - vertices: vertices of the mesh
        - faces: faces of the mesh
        - color: color of the mesh
    '''

    # Create scene
    scene = sp.Scene()
    canvas = scene.create_canvas_3d(width=700, height=700)


    for mesh_data in meshes:
        # create the mesh in the scene
        body_mesh = scene.create_mesh(mesh_data["name"], shared_color=mesh_data["color"])
        # add the first frame of the mesh
        body_mesh.add_mesh_without_normals(mesh_data["vertices"][0], mesh_data["faces"])
        seq_length = mesh_data["vertices"].shape[0]

    if images is None:
        images_canvas_2d = []
    else:
        images_canvas_2d = [scene.create_canvas_2d("image_{}".format(i), width=200, height=200) for i in range(len(images[0]))]

    for idx in range(seq_length):
        update_mesh = []
        labels = []
        for mesh_data in meshes:
            update_mesh.append(scene.update_mesh_positions(mesh_data["name"], mesh_data["vertices"][idx]))
            max_z_idx = mesh_data["vertices"][idx][:, 1].argmin()
            labels.append(["label_"+mesh_data["name"], mesh_data["vertices"][idx][max_z_idx] - np.array([0, 0.1, 0.])])

        # create a frame with the new mesh positions
        frame = canvas.create_frame(meshes=update_mesh)

        for label_data in labels:
            label = scene.create_label(label_data[0], label_data[0])
            frame.add_label(label=label, position=label_data[1])

        for img_idx, img_canvas in enumerate(images_canvas_2d):
            image = scene.create_image()
            image.from_numpy(images[idx][img_idx])
            img_frame = img_canvas.create_frame()
            img_frame.add_image(image, "fit")
            img_frame.add_text(images_names[img_idx], 10, 190, size_in_pixels=16)

        rotation = sp.Transforms.rotation_about_z(np.pi)
        # [-is right, - is up, - is forward]
        scene_camera = sp.Camera([-.0,-4, 2], up_dir=[0,0,1], look_at=[0, 0.5, 0])
        frustums = scene.create_mesh()
        frustums.add_camera_frustum(scene_camera, sp.Colors.Red)
        frame.camera = scene_camera
        frame.add_mesh(frustums)

    scene.link_canvas_events(*([canvas]+images_canvas_2d))

    scene.quantize_updates()
    Title = "Mesh Animation"
    scene.save_as_html(f"{file_name}.html", title=Title)