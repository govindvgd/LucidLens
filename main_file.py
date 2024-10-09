import streamlit as st
from header import NavForAll, set_page_logo
from footer import show_footer
from Running import main as run_main, show_uploaded_image, VideoTransformer
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
import streamlit as st
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

def main():
    set_page_logo("/home/govind/Downloads/MLPR_LucidLens/logo.jpg")
    st.title("Welcome to LucidLens Platform")
    NavForAll()
    st.write("")
    st.write("")
    st.write("")


    option = st.radio("Choose an option:", ("Upload an image", "Capture a photo", "Video input"))

    if option == "Upload an image":
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image_path = "temp_image.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.read())
            show_uploaded_image(image_path)
            mapping_path = "/home/govind/Downloads/MLPR_LucidLens/mapping.pkl"
            caption = run_main(image_path, mapping_path)  # Renamed here
            st.subheader("Generated Caption:")
            st.write(caption)

    elif option == "Capture a photo":
        st.subheader("Capture a photo")
        # Call the function to capture a photo and generate caption
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    elif option == "Video input":
        st.subheader("Video input will take time to process as it depends on computation power.")
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

    show_footer()

def capture_photo_and_generate_caption():
    # Use streamlit_webrtc to continuously capture frames from the webcam
    class FrameCapturer(VideoTransformerBase):
        def transform(self, frame):
            # Display the real-time video feed from the webcam
            cv2.imshow("Webcam", frame.to_ndarray(format="bgr24"))
            # Wait for the user to press the capture key
            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):  # Capture key (press 'c' to capture)
                # Save the captured frame as an image
                image_path = "temp_image.jpg"
                cv2.imwrite(image_path, cv2.cvtColor(frame.to_ndarray(), cv2.COLOR_RGB2BGR))
                cv2.destroyAllWindows()
                # Show the captured image
                show_uploaded_image(image_path)
                # Generate caption for the captured image
                mapping_path = "/home/govind/Downloads/MLPR_LucidLens/mapping.pkl"
                caption = run_main(image_path, mapping_path)
                st.subheader("Generated Caption:")
                st.write(caption)

    # Use webrtc_streamer to capture frames from the webcam
    webrtc_streamer(key="frame-capture", video_transformer_factory=FrameCapturer)



if __name__ == "__main__":
    main()

