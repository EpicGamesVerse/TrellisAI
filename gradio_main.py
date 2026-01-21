import os
import sys
sys.path.append(os.getcwd())


try:  # Try to import xformers, or use the faster one (flash-attn) if they weren't installed
    import xformers  # pyright: ignore[reportMissingImports]
    os.environ['ATTN_BACKEND'] = 'xformers'
except ImportError:
    os.environ['ATTN_BACKEND'] = 'flash-attn'

os.environ['SPCONV_ALGO'] = 'native' 

import torch  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from trellis.pipelines import TrellisImageTo3DPipeline

from version import code_version

# -------------- CMD ARGS PARSE -----------------

# read command-line arguments, passed into this script when launching it:
import argparse
parser = argparse.ArgumentParser(description="Trellis API server")

parser.add_argument("--precision", 
                    choices=["full", "half", "float32", "float16"], 
                    default="full",
                    help="Set the size of variables for pipeline, to save VRAM and gain performance")
parser.add_argument(
    "--share",
    action="store_true",
    help="Create a public Gradio link (requires internet).",
)
cmd_args = parser.parse_args()

# ------------------------------------------------


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)


# ---------------- UI / LOGIC SPLIT ----------------
# The UI is defined in gradio_ui.py and the pipeline logic lives in gradio_logic.py.
# We build the Gradio demo only after the pipeline is initialized.

try:
    import gradio_logic
    import gradio_ui
except ImportError:  # Allows importing via package name if needed
    from trellisai import gradio_logic, gradio_ui  # type: ignore


def create_demo(pipeline: TrellisImageTo3DPipeline):
    ctx = gradio_logic.AppContext(pipeline=pipeline, tmp_dir=TMP_DIR, max_seed=MAX_SEED)
    return gradio_ui.build_demo(ctx)


# Define a function to initialize the pipeline
def initialize_pipeline(precision="full"):
    pipeline = TrellisImageTo3DPipeline.from_pretrained("models")
    # Apply precision settings. Reduce memory usage at the cost of numerical precision:
    print('')
    print(f"used precision: '{precision}'.  Loading...")
    print(f"Trellis repo version {code_version}")
    if precision == "half" or precision=="float16":
        pipeline.to(torch.float16) #cuts memory usage in half
        if "image_cond_model" in pipeline.models:
            pipeline.models['image_cond_model'].half()  #cuts memory usage in half
    # DO NOT MOVE TO CUDA YET. We'll be dynamically loading parts between 'cpu' and 'cuda' soon.
    # Kept for precaution:
    #    pipeline.cuda()
    return pipeline


# Launch the Gradio app
if __name__ == "__main__":
    pipeline = initialize_pipeline(cmd_args.precision)
    print(f'')
    print(f"After launched, open a browser and enter 127.0.0.1:7860 (or whatever IP and port is shown below) into url, as if it was a website:")
    demo = create_demo(pipeline)
    demo.queue()
    demo.launch(inbrowser=True, share=cmd_args.share)
