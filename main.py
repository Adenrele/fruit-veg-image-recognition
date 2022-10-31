import streamlit as st
import io

from torch import load as L, no_grad, topk
from torch import flatten
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image

st.title("Image Classification Demo Model")
st.header("Fruit and vegetable image classification. \n -by Adenrele")
st.text("Upload an image of one of the fruit or veg listed below.")
st.text("Fruit/vegetables included in model training include: \n Apple, Cabbages, Carrots, Cucumbers, Egg plants, Pears and Zucchinis.")

def load_image():
    uploaded_file = st.file_uploader(label='Pick an image to test')
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(65536, 120) #You will obtain error if input layer is incorrect but don't worry, the error will tell you what the number should be. find a way to check shape before building the nn lol. Trial and error is unacceptable.
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 24) #lesson learnt here. Copied code and forgot to change output layer to number of classes. Index error was obtained. Target not in range.

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

classes = ("apple_6", 'apple_braeburn_1', 'apple_crimson_snow_1', 'apple_golden_1', 'apple_golden_2', 'apple_golden_3',
            'apple_granny_smith_1', 'apple_hit_1', 'apple_pink_lady_1', 'apple_red_1', 'apple_red_2', 'apple_red_3',
            'apple_red_delicios_1', 'apple_red_yellow_1', 'apple_rotten_1', 'cabbage_white_1', 'carrot_1', 'cucumber_1',
            'cucumber_3', 'eggplant_violet_1', 'pear_1', 'pear_3', 'zucchini_1', 'zucchini_dark_1'
            )
            

transform = transforms.Compose([transforms.Resize(240),
                            transforms.CenterCrop(270),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
                            ])

def predict(image):
    transformed_image = transform(image)
    for_input = transformed_image.unsqueeze(0)

    net.load_state_dict(L("./fruit_veg.pth"))
    with no_grad():
        output = net(for_input)
    
    probabilities = F.softmax(output[0], dim=0)
    top5_prob, top5_catid = topk(probabilities, 5)
    
    for i in range(top5_prob.size(0)):
        st.write("{}: {:.3f}%".format(classes[top5_catid[i]], top5_prob[i].item()*100))

def main():
    image = load_image()
    result = st.button('Run on image')
    if result:
        st.write('Returning top five results and condifence levels...')
        predict(image)


if __name__ == '__main__':
    main()

#commands streamlit run main.py
#pipreqs
#procfile
#setup.sh
#git init
#Heroku login
#
