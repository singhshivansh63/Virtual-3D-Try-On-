import cv2
import mediapipe as mp
import numpy as np
import pyrender
import trimesh

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# Load the 3D Model
def load_3d_model(file_path):
    mesh = trimesh.load(file_path)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([geom for geom in mesh.geometry.values()])
    return pyrender.Mesh.from_trimesh(mesh)

# Initialize Pyrender Scene and Renderer
scene = pyrender.Scene()
renderer = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)

# Add a camera to the scene
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_pose = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 1.0],
    [0.0, 0.0, 0.0, 1.0],
])
scene.add(camera, pose=camera_pose)

# Load the 3D model and add it to the scene
model_path = r"./assets/necklace.glb"
necklace_3d_model = load_3d_model(model_path)
necklace_node = scene.add(necklace_3d_model)

# Open Webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process frame
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Render the 3D model on the neck
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Get lower jaw landmarks
            jaw_indices = [152, 377, 378, 379, 378, 377, 148]
            jaw_points = [face_landmarks.landmark[i] for i in jaw_indices]
            jaw_coords = np.array([(int(pt.x * w), int(pt.y * h)) for pt in jaw_points])

            # Approximate neck area
            neck_x = int(np.mean(jaw_coords[:, 0]))  # Average X of jawline
            neck_y = int(np.mean(jaw_coords[:, 1])) + 80  # Offset below jawline
            neck_width = int(np.linalg.norm(jaw_coords[0] - jaw_coords[-1]))
            scale_factor = neck_width / 1000  # Adjust scale dynamically

            # Transform the 3D model to align with the neck
            transform = np.eye(4)
            transform[0, 3] = neck_x / w - 0.5  # Normalize X
            transform[1, 3] = -(neck_y / h - 0.5)  # Normalize Y and flip
            transform[2, 3] = -0.5  # Set Z depth
            transform[0, 0] = transform[1, 1] = transform[2, 2] = scale_factor  # Scale

            # Apply the transformation to the necklace node
            scene.set_pose(necklace_node, pose=transform)

            # Render the 3D model
            rendered_image, _ = renderer.render(scene)

            # Overlay the 3D rendered model onto the frame
            rendered_resized = cv2.resize(rendered_image, (frame.shape[1], frame.shape[0]))
            frame = cv2.addWeighted(frame, 0.7, rendered_resized, 0.3, 0)

    # Display the frame
    cv2.imshow("3D Jewelry Try-On", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
face_mesh.close()






            