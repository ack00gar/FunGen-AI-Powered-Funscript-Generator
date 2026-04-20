"""VR dewarp shader bench on a synthetic 8K RGBA texture (offscreen GLFW ctx).

Measures two things separately -- the VR dewarp pipeline's main suspects:

  1. glGenerateMipmap cost on the input texture (called every frame in the
     production path, regardless of whether the current quality level's
     aniso setting actually benefits from mipmaps).
  2. Full VRDewarpShader.render_pass cost at representative quality levels.

Numbers are synced with glFinish before sampling the clock; they are GPU
wall-time, not CPU work. Good enough for gross before/after comparison.
"""
from __future__ import annotations

import time
from typing import Optional, Tuple

import numpy as np

from ..harness import Report, Sample
from ..registry import register


def _try_glfw_ctx(logger=None):
    import glfw
    if not glfw.init():
        raise RuntimeError("glfw.init() failed")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
    win = glfw.create_window(32, 32, "fungen_vr_bench", None, None)
    if not win:
        glfw.terminate()
        raise RuntimeError("glfw.create_window returned NULL (headless GL unavailable?)")
    glfw.make_context_current(win)
    return glfw, win


def _make_input_texture(gl, w: int, h: int) -> int:
    rng = np.random.default_rng(17)
    data = rng.integers(0, 255, (h, w, 4), dtype=np.uint8)
    tex = int(gl.glGenTextures(1))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0,
                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tex


def _make_fbo(gl, w: int, h: int) -> Tuple[int, int]:
    tex = int(gl.glGenTextures(1))
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0,
                    gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    fbo = int(gl.glGenFramebuffers(1))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, fbo)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                              gl.GL_TEXTURE_2D, tex, 0)
    status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
    if status != gl.GL_FRAMEBUFFER_COMPLETE:
        raise RuntimeError(f"FBO incomplete: 0x{status:x}")
    return fbo, tex


def _bench(fn, iters: int, warmup: int, gl) -> list:
    for _ in range(warmup):
        fn()
    gl.glFinish()
    out = []
    for _ in range(iters):
        gl.glFinish()
        t0 = time.perf_counter()
        fn()
        gl.glFinish()
        out.append(time.perf_counter() - t0)
    return out


@register(
    "vr_shader",
    "Offscreen GL bench: glGenerateMipmap + VRDewarpShader.render_pass on synthetic 8K input.",
)
def run(iters: int, warmup: int, **_) -> Report:
    try:
        glfw, win = _try_glfw_ctx()
    except Exception as e:
        r = Report(name="vr_shader",
                   description="skipped - no headless GL context",
                   device="cpu")
        r.extra["error"] = str(e)
        return r

    import OpenGL.GL as gl
    from video.vr_dewarp_shader import VRDewarpShader

    IN_W, IN_H = 8192, 4096      # representative full fisheye / equirect frame
    OUT_W, OUT_H = 2560, 1440    # typical display-side target
    iters = max(20, int(iters))
    warmup = max(3, int(warmup))

    r = Report(
        name="vr_shader",
        description=f"input {IN_W}x{IN_H} RGBA8, output {OUT_W}x{OUT_H}, {iters} iters.",
        device="gpu",
    )

    try:
        # Load the shader.
        import logging as _l
        shader = VRDewarpShader(logger=_l.getLogger("vr_shader_bench"))
        if not shader.compile():
            r.extra["error"] = "shader compile failed"
            return r

        input_tex = _make_input_texture(gl, IN_W, IN_H)
        # Need one mipmap chain to exist before we can measure "regenerate".
        gl.glBindTexture(gl.GL_TEXTURE_2D, input_tex)
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glFinish()

        # --- mipmap cost ---
        def _mipmap():
            gl.glBindTexture(gl.GL_TEXTURE_2D, input_tex)
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

        mm = _bench(_mipmap, iters, warmup, gl)
        r.add(Sample(label=f"glGenerateMipmap {IN_W}x{IN_H}", samples_s=mm, device="gpu"))

        # --- shader render_pass cost at each quality level ---
        # Each level == a (supersample, bicubic, aniso) tuple. Output FBO size
        # tracks L0=2x / L1=2x / L2-L3=1x supersampling.
        fbo1, fbo1_tex = _make_fbo(gl, OUT_W, OUT_H)
        ss_w, ss_h = min(OUT_W * 2, 4096), min(OUT_H * 2, 4096)
        fbo2, fbo2_tex = _make_fbo(gl, ss_w, ss_h)

        levels = [
            ("L0 ss2 bicubic aniso16",   fbo2, ss_w, ss_h, True,  16.0),
            ("L1 ss2 bilinear aniso16",  fbo2, ss_w, ss_h, False, 16.0),
            ("L2 ss1 bilinear aniso4",   fbo1, OUT_W, OUT_H, False, 4.0),
            ("L3 ss1 bilinear aniso1",   fbo1, OUT_W, OUT_H, False, 1.0),
        ]

        # Anisotropic filter extension
        try:
            import OpenGL.GL.EXT.texture_filter_anisotropic as _aniso_ext
            aniso_supported = True
            aniso_cap = float(gl.glGetFloatv(_aniso_ext.GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT))
        except Exception:
            aniso_supported = False
            aniso_cap = 1.0
        r.extra["gpu"] = gl.glGetString(gl.GL_RENDERER).decode(errors="replace")
        r.extra["aniso_cap"] = aniso_cap

        shader_params = dict(
            fisheye_fov_deg=190.0,
            output_fov_deg=90.0,
            yaw_deg=0.0,
            pitch_deg=0.0,
            stereo_format="sbs",
            use_right_eye=False,
            projection="fisheye",
            output_projection="flat",
            output_scale=1.0,
            input_tex_w=IN_W,
            input_tex_h=IN_H,
        )

        for label, fbo, w, h, bicubic, aniso_level in levels:
            if aniso_supported:
                gl.glBindTexture(gl.GL_TEXTURE_2D, input_tex)
                gl.glTexParameterf(gl.GL_TEXTURE_2D,
                                   _aniso_ext.GL_TEXTURE_MAX_ANISOTROPY_EXT,
                                   min(aniso_level, aniso_cap))
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            params = dict(shader_params)
            params["use_bicubic"] = bicubic

            def _pass():
                shader.render_pass(
                    input_texture_id=input_tex,
                    output_fbo=fbo,
                    width=w, height=h,
                    params=params,
                )

            samples = _bench(_pass, iters, warmup, gl)
            r.add(Sample(label=label, samples_s=samples, device="gpu",
                         meta={"w": w, "h": h, "bicubic": bicubic, "aniso": aniso_level}))

        # --- combined: mipmap + render_pass (what every real frame pays) ---
        def _combined():
            gl.glBindTexture(gl.GL_TEXTURE_2D, input_tex)
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            shader.render_pass(
                input_texture_id=input_tex,
                output_fbo=fbo2, width=ss_w, height=ss_h,
                params={**shader_params, "use_bicubic": True},
            )

        comb = _bench(_combined, iters, warmup, gl)
        r.add(Sample(label="combined mipmap + L0 pass",
                     samples_s=comb, device="gpu"))

        # Cleanup
        shader.cleanup()
        gl.glDeleteTextures([input_tex, fbo1_tex, fbo2_tex])
        gl.glDeleteFramebuffers(2, [fbo1, fbo2])
    finally:
        try:
            glfw.destroy_window(win)
        except Exception:
            pass
        try:
            glfw.terminate()
        except Exception:
            pass

    return r
