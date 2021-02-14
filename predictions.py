from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img

from app import IMG_SIZE



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
