import pywavefront
import moderngl
import numpy as np
from PIL import Image
import pyglet

# Configuration
obj_file_path = '/Users/yuanweiyun/Desktop/VR/data/crate.obj'
texture_file_path = '/Users/yuanweiyun/Desktop/VR/data/crate.png'

# Load OBJ file
scene = pywavefront.Wavefront(obj_file_path, create_materials=True, collect_faces=True)

# Pyglet window
window = pyglet.window.Window()

# ModernGL context
ctx = moderngl.create_context()

# Load and flip the texture
texture_image = Image.open(texture_file_path)
texture_data = texture_image.transpose(Image.FLIP_TOP_BOTTOM).convert('RGB').tobytes()
texture = ctx.texture(texture_image.size, 3, texture_data)
texture.build_mipmaps()

# Shader program
prog = ctx.program(
    vertex_shader="""
        #version 330
        in vec3 in_vert;
        in vec2 in_text;
        out vec2 v_text;
        void main() {
            gl_Position = vec4(in_vert, 1.0);
            v_text = in_text;
        }
        """,
    fragment_shader="""
        #version 330
        uniform sampler2D Texture;
        in vec2 v_text;
        out vec4 f_color;
        void main() {
            f_color = texture(Texture, v_text);
        }
        """
)

vertex_data = []
for mesh in scene.mesh_list:
    # Each face contains the indices for a single triangle
    for face in mesh.faces:
        # For each index, we retrieve the vertex, texture coordinate, and normal
        for vertex_index in face:
            # OBJ format starts counting at 1, not 0
            # PyWavefront has already adjusted them to start at 0
            vertex = scene.vertices[vertex_index]
            tex_coord = scene.tex_coords[vertex_index]
            # Flatten the vertex position and texture coordinate into the vertex_data list
            vertex_data.extend(vertex + tex_coord)

# Convert the vertex_data list to a numpy array
vertex_data_np = np.array(vertex_data, dtype='f4')

# Create the buffer and VAO
vbo = ctx.buffer(vertex_data_np.tobytes())
vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_text', stride=20)  # A

@window.event
def on_draw():
    ctx.clear()
    texture.use(location=0)  # Ensure the location matches the sampler2D in the fragment shader
    vao.render(moderngl.TRIANGLES)

    
pyglet.app.run()