from scipy.ndimage import center_of_mass
import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
from scipy.ndimage import center_of_mass

# load model
cnn = tf.keras.models.load_model("digit_cnn_model.keras")


def preprocess(image):

    image = np.array(image)

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    image = 255 - image

    _, image = cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)

    coords = np.column_stack(np.where(image > 0))

    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        digit = image[y_min:y_max, x_min:x_max]
    else:
        digit = image

    digit = cv2.resize(digit, (20,20))

    canvas = np.zeros((28,28))
    canvas[4:24,4:24] = digit

    cy, cx = center_of_mass(canvas)

    if not np.isnan(cx) and not np.isnan(cy):

        shift_x = int(14 - cx)
        shift_y = int(14 - cy)

        M = np.float32([[1,0,shift_x],[0,1,shift_y]])
        canvas = cv2.warpAffine(canvas, M, (28,28))

    canvas = canvas / 255.0
    canvas = canvas.reshape(1,28,28,1)

    return canvas


def predict_digit(data):

    if data is None:
        return "Draw a digit first", {}

    if isinstance(data, dict):
        image = data["composite"]
    else:
        image = data

    image = preprocess(image)

    prediction = cnn.predict(image, verbose=0)[0]

    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    # convert probabilities to dictionary for label display
    prob_dict = {str(i): float(prediction[i]) for i in range(10)}

    return f"Predicted Digit: {digit} | Confidence: {confidence:.2f}%", prob_dict


css = """
body{
background: linear-gradient(135deg,#020024,#090979,#000428);
color:white;
font-family: Arial;
}

h1{
text-align:center;
font-size:40px;
}

.subtitle{
text-align:center;
opacity:0.4;
font-size:14px;
margin-bottom:20px;
}

.gr-button{
background:#00c3ff !important;
color:black !important;
border-radius:10px !important;
}
"""


with gr.Blocks(css=css) as demo:

    gr.Markdown("# Digit Recognizer")
    gr.Markdown("<div class='subtitle'>Made by Harshit</div>")

    gr.Markdown("Draw a number (0–9) on the canvas.")

    canvas = gr.Sketchpad()

    with gr.Row():
        predict_btn = gr.Button("Predict")
        clear_btn = gr.Button("Clear Canvas")

    result = gr.Textbox(label="Prediction")

    probs = gr.Label(num_top_classes=10, label="Digit Probabilities")

    predict_btn.click(
        predict_digit,
        inputs=canvas,
        outputs=[result, probs]
    )

    canvas.change(
        predict_digit,
        inputs=canvas,
        outputs=[result, probs]
    )

    clear_btn.click(
        lambda: (None, "", {}),
        outputs=[canvas, result, probs]
    )

demo.launch()