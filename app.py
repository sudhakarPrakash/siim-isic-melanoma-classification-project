import streamlit as st
from PIL import Image
from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shutil
import random
import time
import os

from utils import menu_contents


def commom_format():
    
    st.write('### Choose options from the sidebar menu to explore more ...')
    st.write('*You are here: 👉  *', '*MENU/{option}*'.format(option=option))
    st.markdown(github_link, unsafe_allow_html=True)
    

def welcome_msg():
    st.write("""
    # Hey, Welcome to this project !!!
        """)
    
    
def about_dataset():
    '''This shows info of dataset used for the project'''
    
    welcome_msg()
    commom_format()
    
    st.write('''## 1️⃣ About this Dataset''')
    st.write(menu_contents['about_dataset'])
    st.write(menu_contents['what_will_it_predict'])
    
    

def show_images(dir_image):
    filenames = os.listdir(dir_image)
    for filename in filenames:
        file_path = os.path.join(dir_image , filename)
        img = Image.open(file_path)
        if img is not None:
            st.write('### '+filename.split('.')[0].upper())
            st.image(img)


def data_preparation_and_exploration():
    
    commom_format()
    st.write('''## 1️⃣ Data Preparation and Exploration''')
    
    st.write('CSV file dataset')
    path_train_csv = os.path.join(base_dir , 'train.csv')
    data = pd.read_csv(path_train_csv)
    st.write(data.head(100))
    
    dir_data_exploration = os.path.join(base_dir,'data_exploration/')
    show_images(dir_data_exploration)
    
            

def model_and_output_files():
    commom_format()
    st.write('''## 1️⃣ Output Files''')
    dir_output_files = os.path.join(base_dir,'output_files/')
    show_images(dir_output_files)


def image_processing():
    commom_format()
    st.write('''## 1️⃣ Image Processing''')
    
    st.write('augmented images')
    path_augmented_images = os.path.join(base_dir , 'image_processing/augmented_images.jpg')
    augmented_image = Image.open(path_augmented_images)
    st.image(augmented_image)
    st.write('code')
    data_augmentation_code = menu_contents['data_augmentation_code']
    with st.echo():
        data_augmentation_code
    dir_image_processing = os.path.join(base_dir,'image_processing/')
    show_images(dir_image_processing)
    
    
    
def sidebar_util():
    '''code to create sidebar to easily navigate from there.
    It sets project title and the thumbnail of the cancer dataset,
    and this sidebar contains a button to predict using model'''
    sidebar = st.sidebar
    thumbnail_path = os.path.join(base_dir,'image_files/thumb76_76.png')
    thumbnail = Image.open(thumbnail_path)
    if thumbnail is not None:
                sidebar.image(
                    thumbnail
                )
    sidebar.title("""
    SIIM-ISIC Melanoma Classification
    Identify melanoma in lesion images""")
    
    global option
    option = sidebar.selectbox(
        ' MENU',
        ['About Dataset','Data Preparation and Exploration','Image Processing','Model and output files'])
    
    if option == 'About Dataset':
        about_dataset()
    if option == 'Data Preparation and Exploration':
        data_preparation_and_exploration()
    if option == 'Model and output files':
        model_and_output_files()
    if option == 'Image Processing':
        image_processing()
    
        
    return sidebar


def generate_result(prediction):
    st.write("""
    ## Predicted RESULT
    """)
    
    for prob in prediction:
        st.text(prob)
    probablity = prediction[0]
    st.write('There is',probablity,'%  chance of malignant')



def predict_result(batches):
    model_path = '/app/siim-isic-melanoma-classification-project/model_and_log_files/model.h5'
    model = load_model(model_path)
    predictions = model.predict(x=batches)
    return predictions



def predict_util():
    '''creates the predict button and file_uploader
    and get the uploaded image '''
    st.write("#  ")
    st.write('# Upload to Predict')
    st.write('Upload here to predict an skin lesion image')
    img_file_buffer = st.file_uploader('choose a skin lesion image',type=['jpg','png','jpeg'])

    try:
        image = Image.open(img_file_buffer)
        image = image.resize((IMG_SIZE,IMG_SIZE))
        img_array = np.array(image)
        st.write("""
            Uploaded Image Preview 224 X 224
            """)
        if image is not None:
            st.image(
                image,
                caption='skin lesion image',
            )
            st.write("""
                Click 'predict' to see results
                """)
            img_array = img_array/255.0
            #test_batch = np.expand_dims(img_array, axis = 0)
            test_batch = img_array.reshape((1, IMG_SIZE, IMG_SIZE, 3))
            prediction = predict_result(test_batch)
            generate_result(prediction)
            #predict_button = st.button('Predict')
            #if predict_button:
            #save uploaded image and get path
            #test_image_path = get_uploaded_image_path(img_array)
            #generate batch of single image
            #st.write('test_image_path = '+test_image_path)
            
            #test_batch = processing_test_file(test_image_path)
            #time.sleep(2)
            #predict on single batch
            #prediction = predict_result(test_batch)
            #generate result from predictions
            #generate_result(prediction)
            #else:
                #st.text('predict button not clicked')
        else:
            st.write('No file uploaded inner..')
        
    except:
        message = 'No file uploaded outer...'
        st.text(message)
        
    


def processing_test_file(test_image_path):
    st.write('processing uploaded file ...')
    try:
        img = load_img(test_image_path, 
                target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img)
        img_array = img_array/255.0
        #test_batch = np.expand_dims(img_array, axis = 0)
        img_array = img_array.reshape((1, IMG_SIZE, IMG_SIZE, 3))
    except:
        st.write('couldn\'t load image')
    return img_array


def get_uploaded_image_path(img_array):
    
    try:
        not_created = True
        while not_created:
            name_of_directory = random.choice(list(range(0, 1885211)))
            try:
                ROOT_DIR = os.path.abspath(os.curdir)
                if str(name_of_directory) not in os.listdir(ROOT_DIR):
                    not_created = False
                    path = ROOT_DIR + "/" + str(name_of_directory)
                    os.mkdir(path)
                    # directory made
            except:
                st.write("""
                    ### some error occured ,Try again !!!
                    """)

        # save image on that directory
        save_img(path+"/test_image.png", img_array)
        image_path = path+"/test_image.png"
    except:
        st.write('something went wrong while saving uploaded file...')
    return image_path


    

def main():
    
    #st.balloons() 
    sidebar = sidebar_util()
    predict_util()
    
    st.write('#  ')
    st.write('#  ')
    st.write("""
    ####  Created By:-     Sudhakar Prakash , Shubham Kumar , Bhanu Ranjan & Aman Saraff
	""")
        
    
    
    
    
if __name__=='__main__':
    IMG_SIZE = 224
    base_dir = '/app/siim-isic-melanoma-classification-project/'
    github_link = "[GitHub link of code](https://github.com/sudhakarPrakash/siim-isic-melanoma-classification-project)"
    
    main()




