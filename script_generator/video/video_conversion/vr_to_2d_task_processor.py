import glfw
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *

from config import RENDER_RESOLUTION, YAW, PITCH
from script_generator.tasks.abstract_task_processor import AbstractTaskProcessor, TaskProcessorTypes
from script_generator.video.video_conversion.opengl.helpers import create_180_dome, render_dome


class VrTo2DTaskProcessor(AbstractTaskProcessor):
    process_type = TaskProcessorTypes.OPENGL

    def task_logic(self):

        # Initialize off-screen GLFW context
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")

        # Create invisible window
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(RENDER_RESOLUTION, RENDER_RESOLUTION, "Offscreen", None, None)
        if not window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")
        glfw.make_context_current(window)

        # OpenGL config
        glEnable(GL_TEXTURE_2D)
        glClearColor(0.0, 0.0, 0.0, 1.0)
        glEnable(GL_DEPTH_TEST)

        # Projection matrix setup
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(90, 1.0, 0.1, 100.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        dome_vertices, dome_tex_coords, dome_indices = create_180_dome()

        # Create texture ID
        texture_id = glGenTextures(1)

        # Create a display list for the dome geometry
        dome_display_list = glGenLists(1)
        glNewList(dome_display_list, GL_COMPILE)
        render_dome(dome_vertices, dome_tex_coords, dome_indices, None)
        glEndList()

        # Render to off-screen buffer
        glViewport(0, 0, RENDER_RESOLUTION, RENDER_RESOLUTION)
        glLoadIdentity()
        # the 3th parameter will zoom in and out to increase / decrease the FOV
        gluLookAt(0, 0, 0.5, 0, 0, -1, 0, 1, 0)
        glRotatef(-YAW, 0, 1, 0)
        glRotatef(-(PITCH + 90), 1, 0, 0)

        glBindTexture(GL_TEXTURE_2D, texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        for task in self.get_task():
            task.start(str(self.process_type))

            # Upload to texture
            h, w, _ = task.preprocessed_frame.shape
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, task.preprocessed_frame)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glCallList(dome_display_list)

            # Read pixels
            pixels = glReadPixels(0, 0, RENDER_RESOLUTION, RENDER_RESOLUTION, GL_RGB, GL_UNSIGNED_BYTE)
            rendered_frame = np.frombuffer(pixels, dtype=np.uint8).reshape(
                RENDER_RESOLUTION, RENDER_RESOLUTION, 3
            )
            rendered_frame = np.flipud(rendered_frame)

            # Store result
            task.rendered_frame = rendered_frame
            task.preprocessed_frame = None

            task.end(str(self.process_type))

            # Debug
            # output_path = os.path.join(DEBUG_PATH, f"frame_{task.id:05d}.png")
            # imageio.imwrite(output_path, rendered_frame)

            self.finish_task(task)

        # Cleanup
        glDeleteLists(dome_display_list, 1)
        glDeleteTextures([texture_id])
        glfw.destroy_window(window)
        glfw.terminate()
