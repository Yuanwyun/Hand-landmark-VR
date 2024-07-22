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
        self.scene_cube = self.load_scene('crate.obj')
        self.scene_marker = self.load_scene('marker.obj')

        # Extract the VAOs from the scene
        self.vao_cube = self.scene_cube.root_nodes[0].mesh.vao.instance(self.prog3d)
        self.vao_marker = self.scene_marker.root_nodes[0].mesh.vao.instance(self.prog3d)

        # Texture of the cube
        self.texture = self.load_texture_2d('crate.png')
        
        # Define the initial position of the virtual object
        # The OpenGL camera is position at the origin, and look at the negative Z axis. The object is at 30 centimeters in front of the camera. 
        self.object_pos = np.array([0.0, 0.0, -30.0])
        
        
        """
        --------------------------------------------------------------------
        TODO: Task 3. 
        Add support to render a rectangle of window size. 
        --------------------------------------------------------------------
        """
        
        
        # Start OpenCV camera 
        self.capture = cv2.VideoCapture(0)
        
        # Get a frame to set the window size and aspect ratio
        ret, frame = self.capture.read() 
        self.aspect_ratio = float(frame.shape[1]) / frame.shape[0]
        self.window_size = (int(720.0 * self.aspect_ratio), 720)

    def render(self, time: float, frame_time: float):
        self.ctx.clear(1.0, 1.0, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)

        """
        ---------------------------------------------------------------
        TODO: Task 3. 
        Get OpenCV video frame, display in OpenGL. 
        Render the frame to a screen-sized rectange. 
        ---------------------------------------------------------------
        """
        
        """
        ---------------------------------------------------------------
        TODO: Task 4.
        Perform hand landmark prediction, and 
        solve PnP to get world landmarks list.
        ---------------------------------------------------------------
        """
        
        # Solve the landmarks in world space
        world_landmarks_list = []
        
        # OpenCV to OpenGL conversion
        # The world points from OpenCV need some changes to be OpenGL ready. 
        # First, the model points are in meters (MediaPipe convention), while our camera matrix is in units. There exists a scale ambiguity of the true hand landmarks, i.e., if we scale up the world points by 1000, its projection remains the same (due to perspective division). 
        # Here we shift the measurement from meter to centimeter, and assume our world space in OpenGL is in centimeters, just for easy visualization and object interaction. So we multiply all points by 100.
        
        # Second, the OpenCV and OpenGL camera coordinate system are different. # OpenCV: right x, down y, into screen z. Image: right x, down y.  
        # OpenGL: right x, up y, out of screen z. Image: right x, up y.
        # Check for image and 3D points flip to make sure the points are properly converted. 
        
        """
        ----------------------------------------------------------------------
        TODO: Task 5.
        We detect a simple pinch gesture, and check if the index finger hits 
        the cube. We approximate by just checking the finger tip is close 
        enough to the cube location.
        ----------------------------------------------------------------------
        """
        grabbed = False
        # It is recommended to work on this task last after all landmarks are in place.
        
        
        """
        ----------------------------------------------------------------------
        TODO: Task 4. 
        Render the markers.
        ----------------------------------------------------------------------
        """
        # Note we have to set the OpenGL projection matrix by following parameters from the OpenCV camera matrix, i.e., the field of view.
        # You can use Matrix44.perspective_projection function, and set the parameters accordingly. Note that the fov must be computed based on the camera matrix. See prediction.py. 
        
        # In this example, a random FOV value is set. Do not use this value in your final program. 
        proj = Matrix44.perspective_projection(45, self.aspect_ratio, 0.1, 1000)
        
        # Translate the object to its position 
        translate = Matrix44.from_translation(self.object_pos)
        
        # Add a bit of random rotation just to be dynamic
        rotate = Matrix44.from_y_rotation(np.sin(time) * 0.5 + 0.2)
        
        # Scale the object up for easy viewing
        scale = Matrix44.from_scale((3, 3, 3))
        
        mvp = proj * translate * rotate * scale
        self.color.value = (1.0, 1.0, 1.0)
        if grabbed: # A bit of feedback when the object is grabbed
            self.color.value = (1.0, 0.0, 0.0)
        self.light.value = (10, 10, 10)
        self.mvp.write(mvp.astype('f4'))
        self.withTexture.value = True
        
        # Render the object
        self.texture.use()
        self.vao_cube.render()
        
        # Render the landmarks
        # ...


if __name__ == '__main__':
    CameraAR.run()
