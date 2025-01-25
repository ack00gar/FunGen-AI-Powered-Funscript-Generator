import numpy as np
from OpenGL.GL import *

def create_180_dome(segments=48):
    """
    Generate vertices, texture coordinates, and indices for a 180° dome.
    """
    vertices, tex_coords, indices = [], [], []

    for i in range(segments + 1):
        theta = np.pi * (i / segments)
        for j in range(segments + 1):
            phi = np.pi * (j / segments)
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            u, v = j / segments, i / segments
            vertices.extend([x, y, z])
            tex_coords.extend([u, v])

    for i in range(segments):
        for j in range(segments):
            p1 = i * (segments + 1) + j
            p2 = p1 + (segments + 1)
            p3 = p1 + 1
            p4 = p2 + 1
            indices.extend([p1, p2, p3, p3, p2, p4])

    return (
        np.array(vertices, dtype=np.float32),
        np.array(tex_coords, dtype=np.float32),
        np.array(indices, dtype=np.uint32),
    )

def render_dome(vertices, tex_coords, indices, texture_id):
    """
    Render the dome using OpenGL.
    If `texture_id` is None, compile only the geometry.
    """
    if texture_id is not None:
        glEnable(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, texture_id)

    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_TEXTURE_COORD_ARRAY)
    glVertexPointer(3, GL_FLOAT, 0, vertices)
    glTexCoordPointer(2, GL_FLOAT, 0, tex_coords)

    glPushMatrix()
    # Flip horizontally for VR
    glScalef(-1.0, 1.0, 1.0)
    glDrawElements(GL_TRIANGLES, len(indices), GL_UNSIGNED_INT, indices)
    glPopMatrix()

    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_TEXTURE_COORD_ARRAY)

    if texture_id is not None:
        glDisable(GL_TEXTURE_2D)