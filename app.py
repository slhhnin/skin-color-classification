import torch
import cv2
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from torchvision import transforms
IMAGE_SHAPE = (224, 224)
classes = ["Dry Skin","Normal Skin", "Oily Skin"]
# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Load the model
@st.cache_resource
def load_model():
    model = torch.load("entire_model_resnet152.pth", map_location=torch.device('cpu'))
    model.eval()
    return model


# Function to preprocess an image from URL or file
def preprocess_image(image):
    if isinstance(image, str):  # Image from URL
        response = requests.get(image)
        img = Image.open(BytesIO(response.content)).convert("RGB")
    else:  # Image from file upload
        img = Image.open(image).convert("RGB")
    read_img = img
    img = transform(img)
    img = img.unsqueeze(0)  # Add batch dimension
    #new_img = read_img.resize((200, 150))

    return img, read_img


# Function to perform prediction
def predict_skin_type(image, model):
    with torch.no_grad():
        prediction = model(image)
        _, predicted_class = prediction.max(1)
        return predicted_class.item()


# Streamlit UI
def main():
    st.title("Human Skin Color Classification")
    st.sidebar.header('Choose how you want to upload a file')
    upload_type = st.sidebar.selectbox('URL or File Upload or WebCam', ('URL', 'File Upload', "WebCam"))

    model = load_model()

    if upload_type == 'URL':
        url = st.text_input("Enter image URL","https://as1.ftcdn.net/v2/jpg/02/98/87/40/1000_F_298874006_MKsJsqZTisaOWhtxRWVsC94oMID5hzkt.jpg")
        if url:
            image , read_img = preprocess_image(url)
            prediction = predict_skin_type(image, model)
            st.write(f"Predicted Skin Type: {classes[prediction]}")

            st.image(read_img, caption="Classifying the Skin", use_column_width=True)
    elif upload_type == "WebCam":
        #st.title("Webcam Display Steamlit App")
        #st.caption("Powered by OpenCV, Streamlit")
        # Set desired webcam resolution (e.g., 640x480)

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 350)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 300)

        frame_placeholder = st.empty()
        prediction_text = st.empty()  # Placeholder for displaying prediction
        stop_button_pressed = st.button("Stop")

        while cap.isOpened() and not stop_button_pressed:
            ret, frame = cap.read()
            if not ret:
                st.write("Video Capture Ended")
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            pil_img = Image.fromarray(frame)
            img = transform(pil_img)
            img = img.unsqueeze(0)  # Add batch dimension
            prediction = predict_skin_type(img, model)
            #st.write(f"Predicted Skin Type: {classes[prediction]}")
            prediction_text.write(f"Predicted Skin Type: {classes[prediction]}", key='prediction')

            frame_placeholder.image(frame, channels="RGB")
            if cv2.waitKey(1) & 0xFF == ord("q") or stop_button_pressed:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        uploaded_file = st.file_uploader("Upload a picture of the skin", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image, read_img = preprocess_image(uploaded_file)
            prediction = predict_skin_type(image, model)
            st.write(f"Predicted Skin Type: {classes[prediction]}")
            st.image(read_img, caption="Classifying the Skin", use_column_width=True)


if __name__ == '__main__':
    main()
