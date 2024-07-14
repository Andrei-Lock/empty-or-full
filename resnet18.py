# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 21:29:55 2022

@author: Andrey
"""

import torch
from torchvision import transforms, models
import torchvision
import streamlit as st
from PIL import Image

class ImageFolderWithPaths(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        path = self.imgs[index][0]
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_image():
    url = st.file_uploader(label="Перетащите или вставьте сюда фотографию пустой или загруженной дороги", type=['jpg', 'png', 'jpeg'])
    return url

def print_predictions(preds):
    st.write(preds)

if __name__ == "__main__":
    example_image = Image.open('./example_image.jpg')
    
    st.write('Определение возможности проезда по дороге по фото')

    image = get_image()

    result = st.button('Распознать тип дороги')
    
    st.write('Beispiel.')
    
    st.image(example_image, caption='Фото-пример')
    
    result_example = st.button('Распознать тип дороги примера')
    
    if result:
        image = Image.open(image)
        model = models.resnet18(weights=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load('resnet18'))
        model.eval()
        
        convert_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        tensor_img = convert_tensor(image)
        
        model.eval()
        
        preds = model(tensor_img.unsqueeze(0))
        pred = torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy()
        
        if pred[0] < 0.5:
            print_predictions('Путь свободен. Результат работы нейросети: ' + str(pred[0]))
        else:
            print_predictions('Путь занят. Результат работы нейросети: ' + str(pred[0]))
    
    if result_example:
        image = example_image
        model = models.resnet18(weights=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 2)
        model.load_state_dict(torch.load('resnet18'))
        model.eval()
        
        convert_tensor = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        tensor_img = convert_tensor(image)
        
        model.eval()
        
        preds = model(tensor_img.unsqueeze(0))
        pred = torch.nn.functional.softmax(preds, dim=1)[:,1].data.cpu().numpy()
        
        if pred[0] < 0.5:
            print_predictions('ПРИМЕР. Путь свободен. Результат работы нейросети: ' + str(pred[0]))
        else:
            print_predictions('ПРИМЕР. Путь занят. Результат работы нейросети: ' + str(pred[0]))
        
