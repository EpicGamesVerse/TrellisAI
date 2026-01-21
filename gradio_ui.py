import os

import gradio as gr  # pyright: ignore[reportMissingImports]
from gradio_litmodel3d import LitModel3D  # pyright: ignore[reportMissingImports]

import gradio_logic


CUSTOM_CSS = """
/* ---------- Meshy-like dark UI ---------- */
:root{
  --bg0:#0b1020;
  --bg1:#0f172a;
  --card:rgba(255,255,255,0.04);
  --card2:rgba(255,255,255,0.06);
  --stroke:rgba(255,255,255,0.10);
  --text:rgba(255,255,255,0.92);
  --muted:rgba(255,255,255,0.70);
  --muted2:rgba(255,255,255,0.55);
  --brand1:#7c3aed;
  --brand2:#22d3ee;
  --brand3:#60a5fa;
}

/* page */
body, .gradio-container{
  background: radial-gradient(1200px 800px at 20% 0%, rgba(124,58,237,0.18), transparent 55%),
              radial-gradient(900px 700px at 85% 10%, rgba(34,211,238,0.12), transparent 60%),
              linear-gradient(180deg, var(--bg0), #070a14 70%) !important;
  color: var(--text) !important;
}

.gradio-container{
  max-width: 1240px !important;
  margin: 0 auto !important;
  padding-top: 14px !important;
}

/* top nav */
#topbar{
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 12px 14px;
  background: rgba(255,255,255,0.03);
  backdrop-filter: blur(8px);
}
#brand{
  display:flex; align-items:center; gap:10px;
}
#brand-badge{
  width: 30px; height: 30px; border-radius: 10px;
  background: linear-gradient(135deg, var(--brand1), var(--brand2));
  box-shadow: 0 8px 22px rgba(124,58,237,0.25);
}
#brand h2{ margin:0; font-size: 16px; letter-spacing: 0.3px;}
#brand span{ color: var(--muted2); font-size: 12px; }
#navlinks{
  display:flex; gap: 10px; justify-content:flex-end; align-items:center;
}
.navpill{
  border: 1px solid var(--stroke);
  background: rgba(255,255,255,0.02);
  padding: 8px 10px;
  border-radius: 999px;
  font-size: 12px;
  color: var(--muted);
}
.navpill b{ color: var(--text); font-weight: 600; }

/* cards */
.card{
  border: 1px solid var(--stroke);
  border-radius: 16px;
  padding: 14px;
  background: var(--card);
  box-shadow: 0 12px 40px rgba(0,0,0,0.35);
}

/* section titles */
.section-title{
  font-size: 12px;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: var(--muted2);
  margin: 0 0 10px 0;
}
.h-title{
  font-size: 20px;
  margin: 0 0 6px 0;
}
.h-sub{
  color: var(--muted);
  margin: 0 0 10px 0;
  font-size: 13px;
  line-height: 1.35;
}

/* inputs */
.gradio-container input, .gradio-container textarea, .gradio-container select{
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  color: var(--text) !important;
  border-radius: 12px !important;
}

label span{
  color: var(--muted) !important;
}

/* primary button */
button.primary, .primary > button{
  background: linear-gradient(135deg, var(--brand1), var(--brand3)) !important;
  border: none !important;
  border-radius: 12px !important;
  box-shadow: 0 10px 30px rgba(124,58,237,0.25);
}
button.primary:hover{ filter: brightness(1.06); transform: translateY(-0.5px); }

/* secondary buttons */
button, .gradio-container .secondary{
  border-radius: 12px !important;
}

/* preview */
#preview video, #viewer canvas{
  border-radius: 14px !important;
  border: 1px solid var(--stroke) !important;
}

/* tabs */
.gradio-container .tabitem{
  border-radius: 12px 12px 0 0 !important;
}
.gradio-container .tabitem.selected{
  border-color: rgba(255,255,255,0.18) !important;
}

/* accordions */
.gradio-container .wrap .accordion{
  border: 1px solid var(--stroke) !important;
  border-radius: 14px !important;
  overflow: hidden;
  background: rgba(255,255,255,0.02) !important;
}
"""


THEME = gr.themes.Soft(primary_hue="violet", secondary_hue="cyan", neutral_hue="slate")


def _set_status(msg: str, visible: bool = True):
    return gr.update(value=msg, visible=visible)


def build_demo(ctx: gradio_logic.AppContext) -> gr.Blocks:
    def _start_session(req: gr.Request):
        return gradio_logic.start_session(ctx, req)

    def _end_session(req: gr.Request):
        return gradio_logic.end_session(ctx, req)

    def _preprocess_image(image):
        return gradio_logic.preprocess_image(ctx, image)

    def _preprocess_images(images):
        return gradio_logic.preprocess_images(ctx, images)

    def _split_image(image):
        return gradio_logic.split_image(ctx, image)

    def _get_seed(randomize_seed: bool, seed: int) -> int:
        return gradio_logic.get_seed(ctx, randomize_seed, seed)

    def _image_to_3d(
        image,
        multiimages,
        is_multiimage,
        seed,
        ss_guidance_strength,
        ss_sampling_steps,
        slat_guidance_strength,
        slat_sampling_steps,
        multiimage_algo,
        req: gr.Request,
      progress=gr.Progress(track_tqdm=True),
    ):
        return gradio_logic.image_to_3d(
            ctx,
            image,
            multiimages,
            is_multiimage,
            seed,
            ss_guidance_strength,
            ss_sampling_steps,
            slat_guidance_strength,
            slat_sampling_steps,
            multiimage_algo,
            req,
        progress,
        )

    def _extract_glb(state, mesh_simplify, texture_size, req: gr.Request, progress=gr.Progress(track_tqdm=True)):
      return gradio_logic.extract_glb(ctx, state, mesh_simplify, texture_size, req, progress)

    def _extract_gaussian(state, req: gr.Request, progress=gr.Progress(track_tqdm=True)):
      return gradio_logic.extract_gaussian(ctx, state, req, progress)

    with gr.Blocks(
        delete_cache=(600, 600),
        theme=THEME,
        css=CUSTOM_CSS,
        title="TRELLIS • Image → 3D",
    ) as demo:
        gr.Markdown(
            """
            <div id="topbar">
              <div style="display:flex; gap:14px; align-items:center; justify-content:space-between;">
                <div id="brand">
                  <div id="brand-badge"></div>
                  <div>
                    <h2>TRELLIS Studio</h2>
                    <span>Meshy‑style Image → 3D • Local</span>
                  </div>
                </div>
                <div id="navlinks">
                  <div class="navpill"><b>Mode</b>: Image → 3D</div>
                  <div class="navpill"><b>Export</b>: GLB / PLY</div>
                  <div class="navpill"><b>Backend</b>: TRELLIS</div>
                </div>
              </div>
            </div>

            <div style="height:12px"></div>

            <div class="card" style="padding:16px 16px 14px 16px;">
              <div class="section-title">Generate</div>
              <div class="h-title">Turn an image into a usable 3D asset</div>
              <div class="h-sub">Upload a single image (or multiple views), generate a mesh + gaussians, then extract a textured GLB. Designed to feel like a modern Meshy-style web app.</div>
              <div style="display:flex; gap:10px; flex-wrap:wrap;">
                <div class="navpill">1) Upload</div>
                <div class="navpill">2) Generate</div>
                <div class="navpill">3) Extract GLB</div>
              </div>
            </div>
            """,
        )

        with gr.Row():
            with gr.Column():
                with gr.Group(elem_classes=["card"]):
                    with gr.Tabs() as input_tabs:
                        with gr.Tab(label="Single Image", id=0) as single_image_input_tab:
                            image_prompt = gr.Image(label="Image Prompt", format="png", image_mode="RGBA", type="pil", height=320)
                        with gr.Tab(label="Multiple Images", id=1) as multiimage_input_tab:
                            multiimage_prompt = gr.Gallery(label="Image Prompt", format="png", type="pil", height=320, columns=3)
                            gr.Markdown(
                                """
                                Input different views of the object in separate images.

                                *NOTE: this is an experimental algorithm without training a specialized model. It may not produce the best results for all images, especially those having different poses or inconsistent details.*
                                """
                            )

                with gr.Group(elem_classes=["card"]):
                    with gr.Accordion(label="Generation Settings", open=False):
                        seed = gr.Slider(0, ctx.max_seed, label="Seed", value=0, step=1)
                        randomize_seed = gr.Checkbox(label="Randomize Seed", value=True)
                        gr.Markdown("Stage 1: Sparse Structure Generation")
                        with gr.Row():
                            ss_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=7.5, step=0.1)
                            ss_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                        gr.Markdown("Stage 2: Structured Latent Generation")
                        with gr.Row():
                            slat_guidance_strength = gr.Slider(0.0, 10.0, label="Guidance Strength", value=3.0, step=0.1)
                            slat_sampling_steps = gr.Slider(1, 50, label="Sampling Steps", value=12, step=1)
                        multiimage_algo = gr.Radio(["stochastic", "multidiffusion"], label="Multi-image Algorithm", value="stochastic")

                    generate_btn = gr.Button("Generate", variant="primary", elem_classes=["primary"])
                    status = gr.Markdown("", visible=False)

                with gr.Group(elem_classes=["card"]):
                    with gr.Accordion(label="GLB Extraction Settings", open=False):
                        mesh_simplify = gr.Slider(0.9, 0.98, label="Simplify", value=0.95, step=0.01)
                        texture_size = gr.Slider(512, 2048, label="Texture Size", value=1024, step=512)

                    with gr.Row():
                        extract_glb_btn = gr.Button("Extract GLB", interactive=False)
                        extract_gs_btn = gr.Button("Extract Gaussian", interactive=False)
                    gr.Markdown(
                        """
                        *NOTE: Gaussian file can be very large (~50MB), it will take a while to display and download.*
                        """
                    )

            with gr.Column():
                with gr.Group(elem_classes=["card"], elem_id="preview"):
                    video_output = gr.Video(label="Generated 3D Asset", autoplay=True, loop=True, height=320)
                with gr.Group(elem_classes=["card"], elem_id="viewer"):
                    model_output = LitModel3D(label="Extracted GLB/Gaussian", exposure=10.0, height=320)

                    with gr.Row():
                        download_glb = gr.DownloadButton(label="Download GLB", interactive=False)
                        download_gs = gr.DownloadButton(label="Download Gaussian", interactive=False)

        is_multiimage = gr.State(False)
        output_buf = gr.State()

        with gr.Row() as single_image_example:
            gr.Examples(
                examples=[
                    f'assets/example_image/{image}'
                    for image in os.listdir("assets/example_image")
                ],
                inputs=[image_prompt],
                fn=_preprocess_image,
                outputs=[image_prompt],
                run_on_click=True,
                examples_per_page=64,
            )

        with gr.Row(visible=False) as multiimage_example:
            gr.Examples(
                examples=gradio_logic.prepare_multi_example(),
                inputs=[image_prompt],
                fn=_split_image,
                outputs=[multiimage_prompt],
                run_on_click=True,
                examples_per_page=8,
            )

        demo.load(_start_session)
        demo.unload(_end_session)

        single_image_input_tab.select(
            lambda: tuple([False, gr.Row.update(visible=True), gr.Row.update(visible=False)]),
            outputs=[is_multiimage, single_image_example, multiimage_example],
        )
        multiimage_input_tab.select(
            lambda: tuple([True, gr.Row.update(visible=False), gr.Row.update(visible=True)]),
            outputs=[is_multiimage, single_image_example, multiimage_example],
        )

        image_prompt.upload(
            _preprocess_image,
            inputs=[image_prompt],
            outputs=[image_prompt],
        )
        multiimage_prompt.upload(
            _preprocess_images,
            inputs=[multiimage_prompt],
            outputs=[multiimage_prompt],
        )

        generate_btn.click(
            _get_seed,
            inputs=[randomize_seed, seed],
            outputs=[seed],
        ).then(
          lambda: (
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
            gr.update(interactive=False),
          ),
          outputs=[generate_btn, extract_glb_btn, extract_gs_btn, download_glb, download_gs],
        ).then(
            lambda: _set_status("Generating… this can take a minute.", True),
            outputs=[status],
        ).then(
            _image_to_3d,
            inputs=[
                image_prompt,
                multiimage_prompt,
                is_multiimage,
                seed,
                ss_guidance_strength,
                ss_sampling_steps,
                slat_guidance_strength,
                slat_sampling_steps,
                multiimage_algo,
            ],
            outputs=[output_buf, video_output],
        ).then(
            lambda: _set_status("Generation complete. You can now extract GLB/Gaussian.", True),
            outputs=[status],
        ).then(
          lambda: (
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
          ),
          outputs=[generate_btn, extract_glb_btn, extract_gs_btn],
        )

        video_output.clear(
            lambda: tuple([_set_status("", False), gr.Button(interactive=False), gr.Button(interactive=False)]),
            outputs=[status, extract_glb_btn, extract_gs_btn],
        )

        extract_glb_btn.click(
            lambda: _set_status("Extracting GLB…", True),
            outputs=[status],
        ).then(
          lambda: (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)),
          outputs=[extract_glb_btn, extract_gs_btn, generate_btn],
        ).then(
            _extract_glb,
            inputs=[output_buf, mesh_simplify, texture_size],
            outputs=[model_output, download_glb],
        ).then(
            lambda: gr.Button(interactive=True),
            outputs=[download_glb],
        ).then(
          lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)),
          outputs=[extract_glb_btn, extract_gs_btn, generate_btn],
        )

        extract_gs_btn.click(
            lambda: _set_status("Extracting Gaussian…", True),
            outputs=[status],
        ).then(
          lambda: (gr.update(interactive=False), gr.update(interactive=False), gr.update(interactive=False)),
          outputs=[extract_glb_btn, extract_gs_btn, generate_btn],
        ).then(
            _extract_gaussian,
            inputs=[output_buf],
            outputs=[model_output, download_gs],
        ).then(
            lambda: gr.Button(interactive=True),
            outputs=[download_gs],
        ).then(
          lambda: (gr.update(interactive=True), gr.update(interactive=True), gr.update(interactive=True)),
          outputs=[extract_glb_btn, extract_gs_btn, generate_btn],
        )

        model_output.clear(
            lambda: tuple([_set_status("", False), gr.Button(interactive=False), gr.Button(interactive=False)]),
            outputs=[status, download_glb, download_gs],
        )

    return demo
    # NOTE: unreachable, kept by mistake

