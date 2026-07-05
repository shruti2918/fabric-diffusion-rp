import runpod, base64, io, sys
from PIL import Image

sys.path.insert(0, '/app/fabric-diffusion')
from pipeline import FabricDiffusionPipeline

print("Loading FabricDiffusion...")
PIPE = FabricDiffusionPipeline(
    "cuda:0",
    texture_checkpoint="/runpod-volume/fabric-diffusion-texture",
    print_checkpoint="/runpod-volume/fabric-diffusion-print",
)
print("Pipeline ready.")

def handler(job):
    try:
        inp     = job["input"]
        img_b64 = inp["image"]
        mode    = inp.get("mode", "texture")

        image = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
        image = image.resize((256, 256), Image.LANCZOS)

        if mode == "print":
            results    = PIPE.flatten_print(image, n_samples=1)
            result_img = results[0].convert("RGB")
        else:
            results    = PIPE.flatten_texture(image, n_samples=1)
            result_img = results[0]

        buf = io.BytesIO()
        result_img.save(buf, format="PNG")
        return {"image": base64.b64encode(buf.getvalue()).decode("utf-8")}

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
