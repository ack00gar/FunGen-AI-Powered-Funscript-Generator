"""GPU dewarp shader for VR display.

One render pass: read mpv's raw fisheye/equirect texture, apply spherical
projection with yaw/pitch/output-FOV uniforms, write a rectilinear texture
that imgui paints. Math from a flatten-fisheye-2.glsl (via the
old gpu_unwarp_worker prototype). Yaw rotation added on top of the
prototype's pitch-only mat3.
"""

from __future__ import annotations

import ctypes
import logging
from typing import Dict, Optional, Tuple

import OpenGL.GL as gl
import numpy as np


_VERT = """
#version 330
in vec2 in_position;
out vec2 v_texcoord;
void main() {
    v_texcoord = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

_FRAG = """
#version 330
#define PI 3.14159265359

uniform sampler2D inputTexture;
uniform float fisheyeFOV;     // radians, source FOV (e.g. 190 deg)
uniform float outputFOV;      // radians, VERTICAL output FOV
uniform float outputAspect;   // outputW / outputH; scales HFOV per-axis
uniform float yaw;            // degrees, view yaw
uniform float pitch;          // degrees, view pitch
uniform int   stereoFormat;   // 0 mono, 1 SBS, 2 TB
uniform int   useRightEye;    // 0 left, 1 right
uniform int   projectionType; // 0 half-equirect, 1 fisheye
uniform int   outputProj;     // 0 rectilinear (flat), 1 stereographic (sg)
uniform float outputScale;    // tuning multiplier on the sg scale (default 1.0)
uniform int   useBicubic;     // 0 bilinear (free), 1 bicubic 4-tap (sharper)
uniform vec2  inputTexSize;   // texel size of inputTexture, for bicubic weights

in vec2 v_texcoord;
out vec4 fragColor;

// Mitchell-Netravali bicubic weights, B=1/3 C=1/3 variant
// (smooth and sharp-ish; avoids the over-sharpening of catmull-rom at ROI edges).
vec4 bicubicSample(sampler2D tex, vec2 uv) {
    vec2 tex_size = 1.0 / inputTexSize;
    vec2 coord = uv / tex_size - 0.5;
    vec2 fxy = fract(coord);
    coord -= fxy;
    // Weights for 4 taps along one axis
    vec4 xcubic = vec4(
        ((-1.0 * fxy.x + 3.0) * fxy.x - 3.0) * fxy.x + 1.0,
        ((3.0 * fxy.x - 6.0) * fxy.x) * fxy.x + 4.0,
        ((-3.0 * fxy.x + 3.0) * fxy.x + 3.0) * fxy.x + 1.0,
        fxy.x * fxy.x * fxy.x) / 6.0;
    vec4 ycubic = vec4(
        ((-1.0 * fxy.y + 3.0) * fxy.y - 3.0) * fxy.y + 1.0,
        ((3.0 * fxy.y - 6.0) * fxy.y) * fxy.y + 4.0,
        ((-3.0 * fxy.y + 3.0) * fxy.y + 3.0) * fxy.y + 1.0,
        fxy.y * fxy.y * fxy.y) / 6.0;
    // Combine 4 bilinear taps via pre-weighted offsets
    vec4 c = coord.xxyy + vec4(-0.5, 1.5, -0.5, 1.5);
    vec4 s = vec4(xcubic.x + xcubic.y, xcubic.z + xcubic.w,
                  ycubic.x + ycubic.y, ycubic.z + ycubic.w);
    vec4 offset = c + vec4(xcubic.y, xcubic.w, ycubic.y, ycubic.w) / s;
    offset *= tex_size.xxyy;
    vec4 s0 = texture(tex, offset.xz);
    vec4 s1 = texture(tex, offset.yz);
    vec4 s2 = texture(tex, offset.xw);
    vec4 s3 = texture(tex, offset.yw);
    float sx = s.x / (s.x + s.y);
    float sy = s.z / (s.z + s.w);
    return mix(mix(s3, s2, sx), mix(s1, s0, sx), sy);
}

mat3 rotationX(float a) {
    float c = cos(a); float s = sin(a);
    return mat3(1.0, 0.0, 0.0,
                0.0, c, -s,
                0.0, s,  c);
}
mat3 rotationY(float a) {
    float c = cos(a); float s = sin(a);
    return mat3( c, 0.0, s,
                0.0, 1.0, 0.0,
                -s, 0.0, c);
}

void main() {
    float xn = v_texcoord.x * 2.0 - 1.0;
    float yn = v_texcoord.y * 2.0 - 1.0;
    vec3 ray;
    if (outputProj == 1) {
        // Stereographic output.
        float scaleV = tan(outputFOV * 0.25) * outputScale;
        float scaleH = scaleV * outputAspect;
        float xp = xn * scaleH;
        float yp = -yn * scaleV;
        float rSq = xp * xp + yp * yp;
        float denom = 1.0 + rSq;
        ray = vec3(2.0 * xp, 2.0 * yp, 1.0 - rSq) / denom;
    } else {
        // Rectilinear (flat) frustum.
        float tanHalfV = tan(outputFOV * 0.5);
        float tanHalfH = tanHalfV * outputAspect;
        ray = normalize(vec3(xn * tanHalfH, -yn * tanHalfV, 1.0));
    }

    // Apply yaw then pitch.
    ray = rotationY(radians(yaw)) * ray;
    ray = rotationX(radians(pitch)) * ray;

    vec2 src;
    if (projectionType == 1) {
        // Fisheye: equidistant projection.
        float p_x = ray.x;
        float p_y = ray.z;
        float p_z = ray.y;
        float p_xz = sqrt(p_x * p_x + p_z * p_z);
        float r = 2.0 * atan(p_xz, p_y) / fisheyeFOV;
        float theta = atan(p_z, p_x);
        src.x = r * cos(theta) * 0.5 + 0.5;
        src.y = 1.0 - (r * sin(theta) * 0.5 + 0.5);
    } else {
        // Half-equirect / equirect: spherical mapping.
        float lon = atan(ray.x, ray.z);
        float lat = asin(clamp(ray.y, -1.0, 1.0));
        src.x = 0.5 + lon / PI;
        src.y = 0.5 - lat / PI;
    }

    // Stereo: pick one eye out of the SBS / TB layout.
    if (stereoFormat == 1) {
        src.x = src.x * 0.5 + (useRightEye == 1 ? 0.5 : 0.0);
    } else if (stereoFormat == 2) {
        src.y = src.y * 0.5 + (useRightEye == 1 ? 0.5 : 0.0);
    }

    src = clamp(src, 0.0, 1.0);
    if (useBicubic == 1) {
        fragColor = bicubicSample(inputTexture, src);
    } else {
        fragColor = texture(inputTexture, src);
    }

    // Interleaved gradient noise dither, ~0.5/255 amplitude. Breaks up
    // 8-bit banding on smooth gradients (skin, sky, walls) without any
    // texture fetch or branch cost. Applied after sampling so the dither
    // is in output space, not source space.
    float ign = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
    fragColor.rgb += (ign - 0.5) / 255.0;
}
"""


_UNIFORM_NAMES = (
    "inputTexture", "fisheyeFOV", "outputFOV", "outputAspect",
    "yaw", "pitch", "stereoFormat", "useRightEye", "projectionType",
    "outputProj", "outputScale", "useBicubic", "inputTexSize",
)


def _compile_one(src: str, kind: int, logger: logging.Logger) -> Optional[int]:
    sid = gl.glCreateShader(kind)
    gl.glShaderSource(sid, src)
    gl.glCompileShader(sid)
    if not gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS):
        log = gl.glGetShaderInfoLog(sid).decode("utf-8", errors="replace")
        kind_name = "vertex" if kind == gl.GL_VERTEX_SHADER else "fragment"
        logger.error(f"VR dewarp {kind_name} shader compile failed:\n{log}")
        gl.glDeleteShader(sid)
        return None
    return int(sid)


class VRDewarpShader:
    """Compile-once + render-many wrapper for the dewarp pass."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.program: Optional[int] = None
        self.uniforms: Dict[str, int] = {}
        self.vao: Optional[int] = None
        self.vbo: Optional[int] = None

    @property
    def is_ready(self) -> bool:
        return self.program is not None and self.vao is not None

    def compile(self) -> bool:
        vs = _compile_one(_VERT, gl.GL_VERTEX_SHADER, self.logger)
        fs = _compile_one(_FRAG, gl.GL_FRAGMENT_SHADER, self.logger)
        if vs is None or fs is None:
            if vs is not None: gl.glDeleteShader(vs)
            if fs is not None: gl.glDeleteShader(fs)
            return False

        prog = gl.glCreateProgram()
        gl.glAttachShader(prog, vs)
        gl.glAttachShader(prog, fs)
        gl.glBindAttribLocation(prog, 0, "in_position")
        gl.glLinkProgram(prog)
        gl.glDeleteShader(vs)
        gl.glDeleteShader(fs)
        if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
            log = gl.glGetProgramInfoLog(prog).decode("utf-8", errors="replace")
            self.logger.error(f"VR dewarp program link failed:\n{log}")
            gl.glDeleteProgram(prog)
            return False

        self.program = int(prog)
        self.uniforms = {
            name: int(gl.glGetUniformLocation(self.program, name))
            for name in _UNIFORM_NAMES
        }

        # Fullscreen triangle (covers the [-1,1] NDC quad with one tri).
        verts = np.array([-1.0, -1.0,  3.0, -1.0,  -1.0,  3.0], dtype=np.float32)
        self.vao = int(gl.glGenVertexArrays(1))
        self.vbo = int(gl.glGenBuffers(1))
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 8, ctypes.c_void_p(0))
        gl.glBindVertexArray(0)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)

        self.logger.debug("VR dewarp shader compiled and ready.")
        return True

    def render_pass(self, *, input_texture_id: int, output_fbo: int,
                    width: int, height: int, params: dict) -> bool:
        """One pass: input_texture_id -> output_fbo at width x height.

        params keys: fisheye_fov_deg, output_fov_deg, yaw_deg, pitch_deg,
        stereo_format ('mono'|'sbs'|'tb'), use_right_eye (bool),
        projection ('fisheye'|'equirect').
        """
        if not self.is_ready:
            return False

        import time as _t
        _t0 = _t.perf_counter()
        try:
            # Save the bits of GL state we touch so imgui's next draw is unaffected.
            # Each glGetIntegerv can force a GPU pipeline sync; skip the active-
            # texture query since imgui uses TEXTURE0 by convention and we reset
            # it to TEXTURE0 on exit. Dropping one query out of five measurably
            # reduces per-frame stall on 8K VR.
            prev_fbo = gl.glGetIntegerv(gl.GL_DRAW_FRAMEBUFFER_BINDING)
            prev_program = gl.glGetIntegerv(gl.GL_CURRENT_PROGRAM)
            prev_vao = gl.glGetIntegerv(gl.GL_VERTEX_ARRAY_BINDING)
            prev_viewport = gl.glGetIntegerv(gl.GL_VIEWPORT)

            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, int(output_fbo))
            gl.glViewport(0, 0, int(width), int(height))
            gl.glUseProgram(self.program)

            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, int(input_texture_id))
            if self.uniforms["inputTexture"] >= 0:
                gl.glUniform1i(self.uniforms["inputTexture"], 0)

            deg2rad = float(np.pi / 180.0)
            if self.uniforms["fisheyeFOV"] >= 0:
                gl.glUniform1f(self.uniforms["fisheyeFOV"],
                               float(params.get("fisheye_fov_deg", 190.0)) * deg2rad)
            if self.uniforms["outputFOV"] >= 0:
                gl.glUniform1f(self.uniforms["outputFOV"],
                               float(params.get("output_fov_deg", 90.0)) * deg2rad)
            if self.uniforms["outputAspect"] >= 0:
                aspect = float(width) / float(height) if height > 0 else 1.0
                gl.glUniform1f(self.uniforms["outputAspect"], aspect)
            if self.uniforms["yaw"] >= 0:
                gl.glUniform1f(self.uniforms["yaw"], float(params.get("yaw_deg", 0.0)))
            if self.uniforms["pitch"] >= 0:
                gl.glUniform1f(self.uniforms["pitch"], float(params.get("pitch_deg", 0.0)))
            stereo = params.get("stereo_format", "mono")
            stereo_enum = 1 if stereo == "sbs" else (2 if stereo == "tb" else 0)
            if self.uniforms["stereoFormat"] >= 0:
                gl.glUniform1i(self.uniforms["stereoFormat"], stereo_enum)
            if self.uniforms["useRightEye"] >= 0:
                gl.glUniform1i(self.uniforms["useRightEye"],
                               1 if params.get("use_right_eye", False) else 0)
            proj_enum = 1 if params.get("projection", "fisheye") == "fisheye" else 0
            if self.uniforms["projectionType"] >= 0:
                gl.glUniform1i(self.uniforms["projectionType"], proj_enum)
            out_proj_enum = 1 if params.get("output_projection", "flat") == "sg" else 0
            if self.uniforms["outputProj"] >= 0:
                gl.glUniform1i(self.uniforms["outputProj"], out_proj_enum)
            out_scale = float(params.get("output_scale", 1.0))
            if self.uniforms["outputScale"] >= 0:
                gl.glUniform1f(self.uniforms["outputScale"], out_scale)
            use_bicubic = 1 if params.get("use_bicubic", False) else 0
            if self.uniforms["useBicubic"] >= 0:
                gl.glUniform1i(self.uniforms["useBicubic"], use_bicubic)
            if self.uniforms["inputTexSize"] >= 0:
                in_w = float(params.get("input_tex_w", width))
                in_h = float(params.get("input_tex_h", height))
                gl.glUniform2f(self.uniforms["inputTexSize"], in_w, in_h)

            gl.glBindVertexArray(self.vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, 3)

            # Restore previous state for the rest of the imgui frame.
            gl.glBindVertexArray(int(prev_vao))
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)  # imgui's default; see save block.
            gl.glUseProgram(int(prev_program))
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, int(prev_fbo))
            gl.glViewport(int(prev_viewport[0]), int(prev_viewport[1]),
                          int(prev_viewport[2]), int(prev_viewport[3]))
            self._last_render_ms = (_t.perf_counter() - _t0) * 1000.0
            self._render_count = getattr(self, '_render_count', 0) + 1
            if self._render_count % 120 == 1:
                self.logger.debug(
                    f"VR dewarp pass {width}x{height} took "
                    f"{self._last_render_ms:.2f}ms (#{self._render_count})")
            return True
        except Exception as e:
            self.logger.warning(f"VR dewarp render pass failed: {e}")
            return False

    def cleanup(self) -> None:
        try:
            if self.vbo is not None:
                gl.glDeleteBuffers(1, [self.vbo])
            if self.vao is not None:
                gl.glDeleteVertexArrays(1, [self.vao])
            if self.program is not None:
                gl.glDeleteProgram(self.program)
        except Exception:
            pass
        self.program = None
        self.uniforms = {}
        self.vao = None
        self.vbo = None
