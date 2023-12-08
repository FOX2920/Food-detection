import os
import numpy as np
import urllib.request
import streamlit as st
from keras.models import load_model
import h5py
from keras.preprocessing import image

# Get the path to the pre-trained model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "base_model_best.h5")

# Load the pre-trained model
with h5py.File(model_path, 'r') as f:
    model = load_model(f)

# Define the list of food classes
classes = [
    'Bánh bèo', 'Bánh bột lọc', 'Bánh căn', 'Bánh canh', 'Bánh chưng', 'Bánh cuốn',
    'Bánh đúc', 'Bánh giò', 'Bánh khọt', 'Bánh mì', 'Bánh pía', 'Bánh tét',
    'Bánh tráng nướng', 'Bánh xèo', 'Bún bò Huế', 'Bún đậu mắm tôm', 'Bún mắm',
    'Bún riêu', 'Bún thịt nướng', 'Cá kho tộ', 'Canh chua', 'Cao lầu',
    'Cháo lòng', 'Cơm tấm', 'Gỏi cuốn', 'Hủ tiếu', 'Mì Quảng', 'Nem chua',
    'Phở', 'Xôi xéo'
]

# Function to preprocess the input image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(300, 300))
    img = image.img_to_array(img) / 255
    img = np.expand_dims(img, axis=0)
    return img

# Main function for the Streamlit app
def main():
    st.title("Nhận Diện Món Ăn Việt Nam")

    # Thanh bên với danh sách các món ăn có thể nhận diện
    st.sidebar.title("Danh Sách Món Ăn có thể Nhận Diện")

    # Thêm dấu '+' trước mỗi món trong danh sách
    formatted_classes = [f"+ {food}" for food in classes]
    st.sidebar.markdown("\n".join(formatted_classes))

    # File uploader and URL input
    uploaded_file = st.file_uploader("Chọn một file", type=["jpg", "jpeg", "png"])
    url = st.text_input('Image Url:', 'https://upload.wikimedia.org/wikipedia/commons/5/53/Pho-Beef-Noodles-2008.jpg')

    if uploaded_file is not None:
        bytes_data = uploaded_file.read()
        st.image(bytes_data, use_column_width=True)
        with open('./test.jpg', 'wb') as f:
            f.write(bytes_data)
    elif url:
        urllib.request.urlretrieve(url, './test.jpg')
        st.markdown(f"<center><img src='{url}' style='width: 90%;'></center>", unsafe_allow_html=True)

    # Preprocess the input image
    img_test = preprocess_image('./test.jpg')

    # Make predictions
    pred_probs = model.predict(img_test)[0]

    # Get the predicted label
    index = np.argmax(pred_probs)
    label = classes[index]

    # Display the predicted label
    st.subheader(f"Dự đoán: {label}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
