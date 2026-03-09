"""Quick Modal script to validate SAM3VideoSemanticPredictor exists in ultralytics."""
import modal

app = modal.App("sam3-validation")

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg", "libgl1")
    .pip_install(
        "ultralytics>=8.3.237",
        "numpy",
    )
    .pip_install(
        "torch==2.4.1+cu121",
        "torchvision==0.19.1+cu121",
        extra_options="--index-url https://download.pytorch.org/whl/cu121",
    )
)


@app.function(image=image, gpu="T4")
def validate_sam3():
    import ultralytics
    print(f"ultralytics version: {ultralytics.__version__}")

    # Check if SAM3VideoSemanticPredictor exists
    try:
        from ultralytics.models.sam import SAM3VideoSemanticPredictor
        print("SUCCESS: SAM3VideoSemanticPredictor found")
        print(f"  class: {SAM3VideoSemanticPredictor}")
    except ImportError as e:
        print(f"FAIL: SAM3VideoSemanticPredictor not importable: {e}")

    # Check what IS in ultralytics.models.sam
    try:
        import ultralytics.models.sam as sam_mod
        attrs = [a for a in dir(sam_mod) if not a.startswith("_")]
        print(f"ultralytics.models.sam exports: {attrs}")
    except Exception as e:
        print(f"Could not inspect sam module: {e}")

    # Check YOLO-World as fallback
    try:
        from ultralytics import YOLO
        model = YOLO("yolov8x-worldv2.pt")
        print(f"SUCCESS: YOLO-World loaded: {type(model)}")
    except Exception as e:
        print(f"YOLO-World load failed: {e}")


@app.local_entrypoint()
def main():
    result = validate_sam3.remote()
    print(result)
