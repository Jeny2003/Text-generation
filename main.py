import gradio as gr
from ultralytics import YOLO
from transformers import pipeline
from PIL import Image

# Load YOLO and FLAN-T5
yolo_model = YOLO("yolov5su.pt")  # or yolov5su.pt
generator = pipeline("text2text-generation", model="google/flan-t5-large")

# Detect objects
def detect_objects(image):
    results = yolo_model(image)
    detections = []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = results[0].names[cls_id]
        confidence = float(box.conf[0])
        detections.append(f"{label} ({confidence:.2f})")
    return detections

# Generate description
def generate_text(prompt, detected_objects):
    object_summary = ', '.join([obj.split(' (')[0] for obj in detected_objects])
    input_text = f"Given these objects: {object_summary}. {prompt}"
    output = generator(
        input_text,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.3
    )
    return output[0]['generated_text']

# Main function for Gradio
def analyze_image(image, prompt):
    if image is None or prompt.strip() == "":
        return "Please upload an image and enter a prompt.", ""

    detected_objects = detect_objects(image)
    if not detected_objects:
        return "No objects detected.", ""

    description = generate_text(prompt, detected_objects)
    return ", ".join(detected_objects), description

# Launch Gradio app
gr.Interface(
    fn=analyze_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Textbox(label="Enter a prompt", placeholder="e.g., Describe the scene in one or two sentences.")
    ],
    outputs=[
        gr.Textbox(label="Detected Objects"),
        gr.Textbox(label="LLM Response")
    ],
    title="Image + Text Generator (YOLO + FLAN-T5)",
    description="Upload an image and enter a prompt. The app detects objects using YOLOv5 and generates a description using the FLAN-T5 model."
).launch()
