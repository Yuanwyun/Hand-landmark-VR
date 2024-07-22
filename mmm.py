import moderngl
import moderngl_window as mglw
from pyrr import Matrix44

import cv2
import numpy as np
import os
from array import array

from prediction import predict, get_camera_matrix, get_fov_y, solvepnp


class CameraAR(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "CameraAR"
    resource_dir = os.path.normpath(os.path.join(__file__, '../data'))
    previousTime = 0
    currentTime = 0
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Shader for rendering 3D objects
        self.prog3d = self.ctx.program(
            vertex_shader='''
                #version 330

                uniform mat4 Mvp;

                in vec3 in_position;
                in vec3 in_normal;
                in vec2 in_texcoord_0;

                out vec3 v_vert;
                out vec3 v_norm;
                out vec2 v_text;

                void main() {
                    gl_Position = Mvp * vec4(in_position, 1.0);
                    v_vert = in_position;
                    v_norm = in_normal;
                    v_text = in_texcoord_0;
                }
            ''',
            fragment_shader='''
                #version 330

                uniform vec3 Color;
                uniform vec3 Light;
                uniform sampler2D Texture;
                uniform bool withTexture;

                in vec3 v_vert;
                in vec3 v_norm;
                in vec2 v_text;

                out vec4 f_color;

                void main() {
                    float lum = clamp(dot(normalize(Light - v_vert), normalize(v_norm)), 0.0, 1.0) * 0.8 + 0.2;
                    if (withTexture) {
                        f_color = vec4(Color * texture(Texture, v_text).rgb * lum, 1.0);
                    } else {
                        f_color = vec4(Color * lum, 1.0);
                    }
                }
            ''',
        )
        self.mvp = self.prog3d['Mvp']
        self.light = self.prog3d['Light']
        self.color = self.prog3d['Color']
        self.withTexture = self.prog3d['withTexture']

        # Load the 3D virtual object, and the marker for hand landmarks
        self.scene_cube = self.load_scene('/Users/yuanweiyun/Desktop/VR/data/crate.obj')
        self.scene_marker = self.load_scene('/Users/yuanweiyun/Desktop/VR/data/marker.obj')

        # Extract the VAOs from the scene
        self.vao_cube = self.scene_cube.root_nodes[0].mesh.vao.instance(self.prog3d)
        self.vao_marker = self.scene_marker.root_nodes[0].mesh.vao.instance(self.prog3d)

        # Texture of the cube
        self.texture = self.load_texture_2d('/Users/yuanweiyun/Desktop/VR/data/crate.png')
        
        # Define the initial position of the virtual object
        # The OpenGL camera is position at the origin, and look at the negative Z axis. The object is at 30 centimeters in front of the camera. 
        self.object_pos = np.array([0.0, 0.0, -30.0])
        
        # Fullscreen quad (two triangles)
        self.quad_vertices = self.ctx.buffer(
            np.array([
             -1.0, -1.0, 0.0, 0.0, 1.0,   # Bottom-left
             1.0, -1.0, 0.0, 1.0, 1.0,    # Bottom-right
             -1.0, 1.0, 0.0, 0.0, 0.0,    # Top-left
             1.0, -1.0, 0.0, 1.0, 1.0,    # Bottom-right
             1.0, 1.0, 0.0, 1.0, 0.0,     # Top-right
             -1.0, 1.0, 0.0, 0.0, 0.0     # Top-left
            ], dtype='f4').tobytes()
)

        self.quad_vao = self.ctx.simple_vertex_array(
        self.prog3d, self.quad_vertices, 'in_position', 'in_texcoord_0'
)
        self.capture = cv2.VideoCapture(0)
        
        # Get a frame to set the window size and aspect ratio
        ret, frame = self.capture.read() 
        self.aspect_ratio = float(frame.shape[1]) / frame.shape[0]
        self.window_size = (int(720.0 * self.aspect_ratio), 720)

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
        # Get a frame from the capture
        ret, frame = self.capture.read()

# Check if frame is read correctly
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return

# Flip the frame vertically and convert to RGB (OpenCV uses BGR by default)
        frame = cv2.flip(frame, 0)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Convert the frame to a texture
        frame_texture = self.ctx.texture(self.window_size, 3, frame.tobytes())
        frame_texture.use()

# Render the full-screen quad with the frame texture
        self.quad_vao.render(moderngl.TRIANGLES)

# Dispose of the texture
        frame_texture.release()



