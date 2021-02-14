from keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img


IMG_SIZE = 224




def generate_result(prediction):
    st.write("""
    ## Predicted RESULT
    """)
    
    probablity = float(prediction[0])
    malignant_prob = probablity*100
    benign_prob = (1-probablity)*100
    st.write('  %.2f' % benign_prob,'%  chance of benign')
    st.write('  %.2f' % malignant_prob,'%  chance of malignant')



def predict_result(batches):
    model_path = '/app/siim-isic-melanoma-classification-project/model_and_log/model.h5'
    model = load_model(model_path)
    predictions = model.predict(x=batches)
    return predictions
