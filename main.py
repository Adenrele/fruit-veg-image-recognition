import streamlit as st
import io
import torch
from PIL import Image
from fruit_veg_classification import net
from fruit_veg_classification import transform
from fruit_veg_classification import classes

st.title("Image Classification Demo Model")
st.header("Fruit and Vegitable image classification \n -by Adenrele")
st.text("Upload an image of one of the fruit or veg listed below.")
st.text("Fruit/vegitables included in model training includes: \n Apple, Cabbages, Carrots, Cucumbers, Egg plants, Pears and Zucchinis.")

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


def predict(image):
    transformed_image = transform(image)
    for_input = transformed_image.unsqueeze(0)

    net.load_state_dict(torch.load("./fruit_veg.pth"))
    with torch.no_grad():
        output = net(for_input)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    for i in range(top5_prob.size(0)):
        st.write("{}: {:.3f}%".format(classes[top5_catid[i]], top5_prob[i].item()*100))

def main():
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Calculating results... \n Returning top 5 results...')
        predict(image)


if __name__ == '__main__':
    main()

#commands streamlit run main.py
#pipreqs
