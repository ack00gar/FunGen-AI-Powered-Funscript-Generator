import imgui
import numpy as np
import ctypes
from OpenGL.GL import *
from OpenGL.GL.shaders import compileProgram, compileShader
import glfw
from PIL import Image
import os

def check_gl_error(operation="Operation"):
    error = glGetError()
    if error != GL_NO_ERROR:
        print(f"OpenGL error after {operation}: {error}")
        return True
    return False

class Simulator3DWindow:
    def __init__(self, app_instance):
        self.app = app_instance
        self.initialized = False
        self.shader = None
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.fbo = None
        self.texture = None
        self.rbo = None
        self.window_size = (400, 500)
        self.logo_texture = None  # OpenGL texture for logo

        self.vertex_shader_source = """
        #version 330 core
        layout (location = 0) in vec3 aPos;
        layout (location = 1) in vec2 aTexCoord;

        out vec3 Normal;
        out vec3 FragPos;
        out vec2 TexCoord;

        uniform float verticalPos;  // 0.0 to 1.0 (bottom to top)
        uniform float rollAngle;    // roll rotation in radians (Z-axis)
        uniform float pitchAngle;   // pitch rotation in radians (X-axis)
        
        void main()
        {
            // Start with original position
            vec3 pos = aPos;
            
            // Scale to larger size for better visibility
            pos *= 0.6;  // Increased from 0.4
            
            // Apply pitch rotation around X-axis first
            float cp = cos(pitchAngle);
            float sp = sin(pitchAngle);
            
            vec3 pitched = vec3(
                pos.x,
                pos.y * cp - pos.z * sp,
                pos.y * sp + pos.z * cp
            );
            
            // Apply roll rotation around Z-axis
            float cr = cos(rollAngle);
            float sr = sin(rollAngle);
            
            vec3 rotated = vec3(
                pitched.x * cr - pitched.y * sr,
                pitched.x * sr + pitched.y * cr,
                pitched.z
            );
            
            // Apply vertical movement: map 0.0-1.0 to -0.4 to +0.4 (reduced amplitude)
            float yOffset = (verticalPos - 0.5) * 0.8;
            rotated.y += yOffset;
            
            // Add viewing angle to see 3D depth properly
            float viewAngleX = 0.3; // ~17 degrees rotation around X axis (look slightly from above)
            float cvx = cos(viewAngleX);
            float svx = sin(viewAngleX);
            
            vec3 angled = vec3(
                rotated.x,
                rotated.y * cvx - rotated.z * svx,
                rotated.y * svx + rotated.z * cvx
            );
            
            // Keep it close to the working depth
            angled.z -= 0.5;
            
            // Pass through the positions for fragment shader
            FragPos = angled;

            // Pass through texture coordinates
            TexCoord = aTexCoord;

            // Calculate proper normal based on position
            // For a cylinder, the normal on the sides points radially outward
            vec3 norm;
            if (abs(aPos.y) > 0.49) {  // Adjusted for height=1.0
                // Top or bottom cap - normal points up or down
                norm = vec3(0.0, sign(aPos.y), 0.0);
            } else {
                // Cylinder side - normal points radially outward from Y axis
                norm = normalize(vec3(aPos.x, 0.0, aPos.z));
            }

            // Transform normal by ALL rotations (pitch, roll, viewing angle)
            // Apply pitch rotation to normal
            vec3 pitchedNorm = vec3(
                norm.x,
                norm.y * cp - norm.z * sp,
                norm.y * sp + norm.z * cp
            );

            // Apply roll rotation to normal
            vec3 rolledNorm = vec3(
                pitchedNorm.x * cr - pitchedNorm.y * sr,
                pitchedNorm.x * sr + pitchedNorm.y * cr,
                pitchedNorm.z
            );

            // Apply viewing angle rotation to normal
            Normal = vec3(
                rolledNorm.x,
                rolledNorm.y * cvx - rolledNorm.z * svx,
                rolledNorm.y * svx + rolledNorm.z * cvx
            );
            
            gl_Position = vec4(angled, 1.0);
        }
        """

        self.fragment_shader_source = """
        #version 330 core
        in vec3 Normal;
        in vec3 FragPos;
        in vec2 TexCoord;

        out vec4 FragColor;

        uniform sampler2D logoTexture;
        uniform bool useTexture;

        void main()
        {
            vec3 baseColor;

            // Check if this is a cap vertex by looking at UV coordinates
            // Center vertices of caps have UV = (-1, -1), making nearby fragments identifiable
            bool isCap = (TexCoord.x < -0.5 && TexCoord.y < -0.5) || abs(Normal.y) > 0.6;

            if (isCap) {
                // Top and bottom caps - lighter blue/cyan with darker edges
                // Calculate distance from cap center for edge darkening
                float distFromCenter = length(vec2(TexCoord.x + 1.0, TexCoord.y + 1.0));
                float edgeFactor = smoothstep(0.8, 1.2, distFromCenter);
                vec3 capColor = vec3(0.5, 0.8, 1.0);  // Light cyan
                vec3 edgeColor = vec3(0.2, 0.3, 0.5);  // Dark blue edge
                baseColor = mix(capColor, edgeColor, edgeFactor * 0.6);
            } else {
                // Cylinder body - use logo texture on front portion, uniform blue elsewhere
                if (useTexture &&
                    TexCoord.x >= 0.0 && TexCoord.x <= 1.0 &&
                    TexCoord.y >= 0.0 && TexCoord.y <= 1.0) {
                    // Sample the logo texture for front center of cylinder
                    vec4 texColor = texture(logoTexture, TexCoord);
                    // Use texture color with slight blue tint
                    baseColor = mix(texColor.rgb, vec3(0.3, 0.5, 0.9), 0.2);
                } else {
                    // Rest of cylinder body - uniform medium blue
                    baseColor = vec3(0.3, 0.5, 0.9);
                }
            }

            // Simplified uniform lighting to avoid color banding
            vec3 lightDir = normalize(vec3(0.5, 0.8, 1.0));

            // Gentle diffuse lighting
            float diff = max(dot(normalize(Normal), lightDir), 0.0) * 0.3;

            // Uniform ambient with subtle diffuse
            vec3 ambient = baseColor * 0.75;
            vec3 diffuse = baseColor * diff;

            vec3 finalColor = ambient + diffuse;

            FragColor = vec4(finalColor, 1.0);
        }
        """

        # Create cylinder geometry
        self.vertices, self.indices = self.create_cylinder()

    def create_cylinder(self):
        """Create a vertical cylinder"""
        vertices = []
        indices = []

        # Cylinder parameters
        radius = 0.35  # Increased from 0.25
        height = 1.0   # Increased from 0.8
        segments = 20  # Number of segments around the cylinder for smoothness

        # Generate cylinder vertices (standing vertically)
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)

            # UV coordinates for texture mapping
            # Logo appears only on front-center portion (25% of circumference, 50% of height)
            # Center the logo at i/segments = 0.25 (front facing viewer after rotation)
            # Map a range around 0.25 to UV [0,1], rest maps outside to show blue color

            # U: Map front 25% of circumference (centered at 0.25) to [0, 1]
            u_normalized = i / segments
            # Center around 0.25, with logo spanning ±0.125 (total 0.25 or 25%)
            u = (u_normalized - 0.125) / 0.25  # Maps [0.125, 0.375] to [0, 1]

            # V: Map middle 50% of height to [0, 1]
            # V should be 0 at top, 1 at bottom (flipped for correct orientation)
            # Middle 50% means we map the range [0.25, 0.75] of height to [0, 1]
            v_pos = 0.5  # This will be adjusted per vertex below

            # Bottom circle vertex (y = -height/2)
            v_bottom = (0.75 - 0.25) / 0.5 + 0  # Maps to V=1.0 (but we'll calculate properly)
            v_bottom = 1.0 + (0.25 / 0.5)  # Bottom is at 75% down -> V > 1 (outside logo)
            vertices.extend([x, -height/2, z, u, 1.5])  # V=1.5, outside logo range

            # Top circle vertex (y = +height/2)
            v_top = 0 - (0.25 / 0.5)  # Top is at 25% down -> V < 0 (outside logo)
            vertices.extend([x, height/2, z, u, -0.5])  # V=-0.5, outside logo range
        
        # Add center vertices for caps (with UV coordinates outside [0,1] to prevent logo sampling)
        bottom_center_idx = len(vertices) // 5  # Now 5 floats per vertex (x,y,z,u,v)
        vertices.extend([0.0, -height/2, 0.0, -1.0, -1.0])  # Bottom center - UV outside range
        top_center_idx = len(vertices) // 5
        vertices.extend([0.0, height/2, 0.0, -1.0, -1.0])   # Top center - UV outside range
        
        # Generate cylinder side triangles
        for i in range(segments):
            next_i = (i + 1) % segments
            
            # Each quad on the side needs 2 triangles
            # Triangle 1
            indices.extend([i*2, i*2+1, next_i*2+1])
            # Triangle 2
            indices.extend([i*2, next_i*2+1, next_i*2])
        
        # Generate bottom cap triangles (counter-clockwise when viewed from below)
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.extend([bottom_center_idx, i*2, next_i*2])
        
        # Generate top cap triangles (counter-clockwise when viewed from above)
        for i in range(segments):
            next_i = (i + 1) % segments
            indices.extend([top_center_idx, next_i*2+1, i*2+1])

        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)

    def init_opengl(self):
        try:
            # Create and bind VAO
            self.vao = glGenVertexArrays(1)
            glBindVertexArray(self.vao)
            check_gl_error("VAO creation and binding")
            
            # Compile shaders
            vertex_shader = compileShader(self.vertex_shader_source, GL_VERTEX_SHADER)
            fragment_shader = compileShader(self.fragment_shader_source, GL_FRAGMENT_SHADER)
            check_gl_error("Shader compilation")

            # Create program and attach shaders
            self.shader = glCreateProgram()
            glAttachShader(self.shader, vertex_shader)
            glAttachShader(self.shader, fragment_shader)
            glLinkProgram(self.shader)
            check_gl_error("Shader program creation and linking")

            # Check for linking errors
            if not glGetProgramiv(self.shader, GL_LINK_STATUS):
                error_log = glGetProgramInfoLog(self.shader)
                print(f"❌ Shader linking failed: {error_log}")
                return

            glDeleteShader(vertex_shader)
            glDeleteShader(fragment_shader)
            check_gl_error("Shader deletion")

            # Create and bind VBO
            self.vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo)
            glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
            check_gl_error("VBO creation and data upload")

            # Create and bind EBO
            self.ebo = glGenBuffers(1)
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
            check_gl_error("EBO creation and data upload")

            # Set up vertex attributes
            # Vertex format: x, y, z, u, v (5 floats = 20 bytes per vertex)
            stride = 20  # 5 floats * 4 bytes

            # Position attribute (location = 0) - 3 floats at offset 0
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)

            # Texture coordinate attribute (location = 1) - 2 floats at offset 12
            glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, stride, ctypes.c_void_p(12))
            glEnableVertexAttribArray(1)

            check_gl_error("Setting up vertex attributes")

            # Load logo texture
            try:
                # Get the path to assets/logo.png
                script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                logo_path = os.path.join(script_dir, "assets", "logo.png")

                if os.path.exists(logo_path):
                    # Load image with PIL
                    img = Image.open(logo_path)
                    img = img.convert("RGBA")  # Ensure RGBA format
                    img_data = np.array(img, dtype=np.uint8)

                    # Create OpenGL texture
                    self.logo_texture = glGenTextures(1)
                    glBindTexture(GL_TEXTURE_2D, self.logo_texture)

                    # Upload texture data
                    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, img.width, img.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

                    # Set texture parameters
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

                    glBindTexture(GL_TEXTURE_2D, 0)
                    check_gl_error("Logo texture loading")
                    print(f"✓ Loaded logo texture from {logo_path}")
                else:
                    print(f"⚠ Logo not found at {logo_path}, using default colors")
            except Exception as e:
                print(f"⚠ Failed to load logo texture: {e}")

            # Create framebuffer for imgui rendering
            self.fbo = glGenFramebuffers(1)
            glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
            check_gl_error("Framebuffer creation and binding")

            # Create texture for framebuffer
            self.texture = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            # Use GL_RGB format
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.window_size[0], self.window_size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.texture, 0)
            check_gl_error("Texture creation and setup for framebuffer")

            # Create renderbuffer for depth
            self.rbo = glGenRenderbuffers(1)
            glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.window_size[0], self.window_size[1])
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, self.rbo)
            check_gl_error("Renderbuffer setup")

            # Check framebuffer status
            if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                print("❌ Framebuffer is not complete!")
                check_gl_error("Framebuffer completeness check")

            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            check_gl_error("Unbinding framebuffer")

            self.initialized = True
            self._shader_version = 18  # v18: Logo restored + simplified uniform lighting
        except Exception as e:
            print(f"OpenGL Initialization Error: {e}")
            self.initialized = False

    def render(self):
        app_state = self.app.app_state_ui
        if not app_state.show_simulator_3d:
            return

        # Initialize debug counter at the start of render
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0

        imgui.set_next_window_size(self.window_size[0], self.window_size[1], condition=imgui.ONCE)
        # Remove scrollbars
        visible, opened_state = imgui.begin("3D Simulator", closable=True, flags=imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE)

        # Update app state based on window close button
        if app_state.show_simulator_3d != opened_state:
            app_state.show_simulator_3d = opened_state

        # Early exit if window is not visible - skip expensive OpenGL operations
        if not visible:
            imgui.end()
            return

        # Start performance monitoring for 3D simulator rendering
        perf_start_time = None
        if hasattr(self.app, 'gui_instance') and hasattr(self.app.gui_instance, 'component_render_times'):
            import time
            perf_start_time = time.perf_counter()

        window_size = imgui.get_window_size()
        if self.window_size != (window_size.x, window_size.y) and window_size.x > 0 and window_size.y > 0:
            self.window_size = (int(window_size.x), int(window_size.y))
            if self.initialized:
                # Resize framebuffer
                glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
                
                glBindTexture(GL_TEXTURE_2D, self.texture)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, self.window_size[0], self.window_size[1], 0, GL_RGB, GL_UNSIGNED_BYTE, None)
                
                glBindRenderbuffer(GL_RENDERBUFFER, self.rbo)
                glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, self.window_size[0], self.window_size[1])

                # Check if framebuffer is still complete after resize
                if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
                    print("Framebuffer is not complete after resize!")
                    check_gl_error("Framebuffer resize")
                
                glBindFramebuffer(GL_FRAMEBUFFER, 0)


        if not self.initialized:
            self.init_opengl()
        elif not hasattr(self, '_shader_version') or self._shader_version != 18:
            # Force recompilation of shaders (v18: logo + uniform lighting)
            self.init_opengl()
            self._shader_version = 18

        if not self.initialized:
            imgui.text("Failed to initialize OpenGL")
            imgui.text("Check console for error messages")
            imgui.end()
            return

        # TEST: Recreate framebuffer on every frame to avoid state issues
        if hasattr(self, 'test_fbo'):
            glDeleteFramebuffers(1, [self.test_fbo])
        if hasattr(self, 'test_texture'):
            glDeleteTextures(1, [self.test_texture])
            
        # Create new framebuffer
        self.test_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.test_fbo)
        
        # Create new texture
        self.test_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.test_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, self.window_size[0], self.window_size[1], 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.test_texture, 0)

        # Check status
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("❌ Test framebuffer incomplete!")

        check_gl_error("Creating fresh framebuffer")

        glViewport(0, 0, self.window_size[0], self.window_size[1])
        check_gl_error("Setting viewport")

        glClearColor(0.0, 0.0, 0.0, 0.0)  # Transparent black background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        check_gl_error("Clearing buffers")

        # Enable alpha blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)  # Counter-clockwise winding is front-facing (OpenGL default)

        # Ensure solid fill mode (no wireframe)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        check_gl_error("Setting up rendering mode")

        glUseProgram(self.shader)
        check_gl_error("Using shader program")

        # Get funscript positions
        primary_pos = getattr(app_state, 'gauge_value_t1', 50)  # 0-100 (up/down)
        secondary_pos = getattr(app_state, 'gauge_value_t2', 50)  # 0-100 (roll)
        
        # Check for third axis (when available in future)
        tertiary_pos = getattr(app_state, 'gauge_value_t3', None)  # Third axis (pitch)
        if tertiary_pos is None:
            tertiary_pos = 50  # No pitch movement when third axis not available
        
        # Convert to shader values
        vertical_pos = primary_pos / 100.0  # 0.0 to 1.0
        roll_angle = np.radians((secondary_pos - 50) * 0.9)  # ±45° max roll
        pitch_angle = np.radians((tertiary_pos - 50) * 0.6) if tertiary_pos != 50 else 0.0  # ±30° max pitch
        
        # Update debug frame counter
        if not hasattr(self, '_debug_frame_count'):
            self._debug_frame_count = 0
        self._debug_frame_count += 1

        # Set shader uniforms
        vertical_loc = glGetUniformLocation(self.shader, "verticalPos")
        roll_loc = glGetUniformLocation(self.shader, "rollAngle")
        pitch_loc = glGetUniformLocation(self.shader, "pitchAngle")
        
        if vertical_loc != -1:
            glUniform1f(vertical_loc, vertical_pos)
            
        if roll_loc != -1:
            glUniform1f(roll_loc, roll_angle)
            
        if pitch_loc != -1:
            glUniform1f(pitch_loc, pitch_angle)

        # Bind logo texture if available and enabled in settings
        logo_enabled = self.app.app_settings.get('show_3d_simulator_logo', True)
        if self.logo_texture is not None and logo_enabled:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.logo_texture)

            # Set texture uniform to use texture unit 0
            texture_loc = glGetUniformLocation(self.shader, "logoTexture")
            if texture_loc != -1:
                glUniform1i(texture_loc, 0)

            # Enable texture flag
            use_texture_loc = glGetUniformLocation(self.shader, "useTexture")
            if use_texture_loc != -1:
                glUniform1i(use_texture_loc, 1)
        else:
            # Disable texture flag
            use_texture_loc = glGetUniformLocation(self.shader, "useTexture")
            if use_texture_loc != -1:
                glUniform1i(use_texture_loc, 0)

        glBindVertexArray(self.vao)
        check_gl_error("Binding VAO")
        
        # Draw the 3D cuboid
        num_indices = len(self.indices)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
        
        glBindVertexArray(0)
        check_gl_error("Unbinding VAO")


        # Unbind framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        check_gl_error("Unbinding framebuffer")

        # Display the rendered texture in ImGui
        # UV coordinates: (0,1) to (1,0) flips the Y axis since OpenGL and ImGui have different Y directions
        if hasattr(self, 'test_texture'):
            imgui.image(self.test_texture, self.window_size[0], self.window_size[1], (0, 1), (1, 0))
            check_gl_error("Displaying test texture in ImGui")
        else:
            imgui.text("No texture to display")

        # End performance monitoring and record the timing
        if perf_start_time is not None and hasattr(self.app, 'gui_instance'):
            render_time_ms = (time.perf_counter() - perf_start_time) * 1000
            self.app.gui_instance.component_render_times["3D_Simulator"] = render_time_ms

        imgui.end()

    def render_3d_content(self, width=None, height=None):
        """
        Render 3D simulator content without window wrapper.
        Can be called from standalone window or overlay.

        Args:
            width: Override window width (defaults to self.window_size[0])
            height: Override window height (defaults to self.window_size[1])
        """
        app_state = self.app.app_state_ui

        # Use provided dimensions or default to window size
        if width is None or height is None:
            width = self.window_size[0]
            height = self.window_size[1]

        # Update window size if different
        if (width, height) != self.window_size and width > 0 and height > 0:
            self.window_size = (int(width), int(height))

        # Initialize OpenGL if needed
        if not self.initialized:
            self.init_opengl()
        elif not hasattr(self, '_shader_version') or self._shader_version != 18:
            self.init_opengl()
            self._shader_version = 18

        if not self.initialized:
            imgui.text("Failed to initialize OpenGL")
            imgui.text("Check console for error messages")
            return

        # Create fresh framebuffer for this frame
        if hasattr(self, 'test_fbo'):
            glDeleteFramebuffers(1, [self.test_fbo])
        if hasattr(self, 'test_texture'):
            glDeleteTextures(1, [self.test_texture])

        self.test_fbo = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.test_fbo)

        self.test_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.test_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.test_texture, 0)

        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            imgui.text("Framebuffer incomplete!")
            glBindFramebuffer(GL_FRAMEBUFFER, 0)
            return

        check_gl_error("Creating fresh framebuffer")

        # Set up rendering
        glViewport(0, 0, width, height)
        glClearColor(0.0, 0.0, 0.0, 0.0)  # Transparent black background
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Enable alpha blending for transparency
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)  # Counter-clockwise winding is front-facing (OpenGL default)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)

        glUseProgram(self.shader)

        # Get funscript positions
        primary_pos = getattr(app_state, 'gauge_value_t1', 50)
        secondary_pos = getattr(app_state, 'gauge_value_t2', 50)
        tertiary_pos = getattr(app_state, 'gauge_value_t3', None)
        if tertiary_pos is None:
            tertiary_pos = 50

        # Convert to shader values
        vertical_pos = primary_pos / 100.0
        roll_angle = np.radians((secondary_pos - 50) * 0.9)
        pitch_angle = np.radians((tertiary_pos - 50) * 0.6) if tertiary_pos != 50 else 0.0

        # Set shader uniforms
        vertical_loc = glGetUniformLocation(self.shader, "verticalPos")
        roll_loc = glGetUniformLocation(self.shader, "rollAngle")
        pitch_loc = glGetUniformLocation(self.shader, "pitchAngle")

        if vertical_loc != -1:
            glUniform1f(vertical_loc, vertical_pos)
        if roll_loc != -1:
            glUniform1f(roll_loc, roll_angle)
        if pitch_loc != -1:
            glUniform1f(pitch_loc, pitch_angle)

        # Bind logo texture if available and enabled in settings
        logo_enabled = self.app.app_settings.get('show_3d_simulator_logo', True)
        if self.logo_texture is not None and logo_enabled:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.logo_texture)

            # Set texture uniform to use texture unit 0
            texture_loc = glGetUniformLocation(self.shader, "logoTexture")
            if texture_loc != -1:
                glUniform1i(texture_loc, 0)

            # Enable texture flag
            use_texture_loc = glGetUniformLocation(self.shader, "useTexture")
            if use_texture_loc != -1:
                glUniform1i(use_texture_loc, 1)
        else:
            # Disable texture flag
            use_texture_loc = glGetUniformLocation(self.shader, "useTexture")
            if use_texture_loc != -1:
                glUniform1i(use_texture_loc, 0)

        # Draw the 3D cuboid
        glBindVertexArray(self.vao)
        num_indices = len(self.indices)
        glDrawElements(GL_TRIANGLES, num_indices, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        # Unbind framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Display the rendered texture
        if hasattr(self, 'test_texture'):
            imgui.image(self.test_texture, width, height, (0, 1), (1, 0))
        else:
            imgui.text("No texture to display")

    def translate(self, matrix, x, y, z):
        translation = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        return matrix @ translation

    def rotate(self, matrix, angle, x, y, z):
        c, s = np.cos(angle), np.sin(angle)
        t = 1 - c
        rotation = np.array([
            [t*x*x + c,   t*x*y - s*z, t*x*z + s*y, 0],
            [t*x*y + s*z, t*y*y + c,   t*y*z - s*x, 0],
            [t*x*z - s*y, t*y*z + s*x, t*z*z + c,   0],
            [0,           0,           0,           1]
        ], dtype=np.float32)
        return matrix @ rotation

    def perspective(self, fov, aspect, near, far):
        f = 1.0 / np.tan(fov / 2.0)
        return np.array([
            [f / aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
            [0, 0, -1, 0]
        ], dtype=np.float32)

