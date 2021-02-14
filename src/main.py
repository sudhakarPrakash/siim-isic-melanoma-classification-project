import sys

import os
import shutil
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib.image as mpimg
import sklearn
import matplotlib
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential , load_model
from keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,CSVLogger
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy,binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
from sklearn.utils import class_weight
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.metrics import roc_curve,auc,roc_auc_score


def python_version():
    return sys.version

def library_version():
    print('{:<15} {}'.format('Pandas',pd.__version__))
    print('{:<15} {}'.format('NumPy',np.__version__))
    print('{:<15} {}'.format('Matplotlib',matplotlib.__version__))
    print('{:<15} {}'.format('Sklearn',sklearn.__version__))
    print('{:<15} {}'.format('Tensorflow',tf.__version__))
    print('{:<15} {}'.format('Keras',keras.__version__))


def prepare_data():
    '''explore dataset csv file and handles 
    missing value and returns missed informatiom
    in the form of a dataset'''
    data = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
    total_patients = len(data["patient_id"].unique())
    total_records = data.shape[0]
    no_males = len(data[data["sex"]=="male"])
    no_females = len(data[data["sex"]=="female"])
    
    missed_data = []
    for column in data.columns:
        no_missed_values = data[column].isna().sum()
        percentage = round((no_missed_values/total_records)*100,2)
        missed_data.append([no_missed_values,percentage])
        
    missed_df=pd.DataFrame(index=data.columns,columns=['no_of_missing_values','missing_data_percentage'],data=missed_data)
    missed_df.style.apply(lambda x: ['background: lightgreen' if x.missing_data_percentage>0 else '' for i in x],axis=1)
    
    data['sex'] = data['sex'].fillna('male')
    data['age_approx'] = data['age_approx'].fillna(data['age_approx'].mode())
    return missed_df


def plot_age_distribution():
    ax = data['age_approx'].value_counts().sort_index().plot(kind='bar',fontsize=14)
    ax.set_title("patient's age distribution",fontsize=16)
    ax.set_xlabel('age_approx',fontsize=14)
    ax.set_ylabel('counts',fontsize=14)
    plt.tight_layout()
    plt.savefig('age_distribution.jpg')


def plot_age_wrt_gender():
    # KDE plot of age that were diagnosed as benign
    sns.kdeplot(data.loc[data['sex'] == 'male', 'age_approx'],color='g',label = 'Male',shade=True)

    # KDE plot of age that were diagnosed as malignant
    sns.kdeplot(data.loc[data['sex'] == 'female', 'age_approx'],color='b',label = 'Female',shade=True)

    # Labeling of plot
    plt.title('Distribution of Ages wrt gender')
    plt.xlabel('Age (years)')
    plt.ylabel('Density')
    plt.savefig('Distribution of Ages wrt gender')
    
    
def benign_malignant_distribution():
    ax = data['benign_malignant'].value_counts().plot(kind='bar',fontsize=14)
    ax.set_title('benigin v/s malignant Distribution',fontsize=16)
    ax.set_ylabel('counts',fontsize=14)
    ax.set_xticklabels(data['benign_malignant'].unique(),rotation=25,fontsize=12)
    for i in ax.patches:
    #     ax.text(i.get_x(), i.get_height(), str(i.get_height()),ha='center',va='bottom',rotation=45)
        ax.text(i.get_x() + i.get_width()/2.,i.get_height(),
                    '%d' % int(i.get_height()),
                    ha='center', va='bottom',fontsize=12)
        
    plt.tight_layout()
    plt.savefig('Benign_malignant_distribution.jpg')
    
    
def target_distribution():
    fig = plt.figure(figsize=(8,4))
    ax = data['target'].value_counts().plot(kind='bar',width=0.25)
    ax.set_title('Target',fontsize=16)
    ax.set_ylabel('counts',fontsize=14)
    ax.set_xticklabels(data['target'].unique(),rotation=0,fontsize=12)

    for i in ax.patches:
    #     ax.text(i.get_width(), i.get_y(), str(i.get_width()))
        ax.text(i.get_x() + i.get_width()/2.,i.get_height(),
                    '%d' % int(i.get_height()),
                    ha='center', va='bottom',fontsize=12)

    plt.tight_layout()
    plt.savefig('Target_distribution.jpg')
    
    
def plot_age_wrt_target():
    # KDE plot of age that were diagnosed as benign
    sns.kdeplot(data.loc[data['target'] == 0, 'age_approx'], color='green',label = 'Benign',shade=True)

    # KDE plot of age that were diagnosed as malignant
    sns.kdeplot(data.loc[data['target'] == 1, 'age_approx'], color='blue',label = 'Malignant',shade=True)

    # Labeling of plot
    plt.title('Distribution of Ages')
    plt.xlabel('Age (years)')
    plt.ylabel('Density')
    
    
def plot_sex_wrt_benign_malignant():
    male_with_benign = len(data[(data['sex']=='male') & (data['benign_malignant']=='benign')])
    male_with_malignant = len(data[(data['sex']=='male') & (data['benign_malignant']=='malignant')])
    female_with_benign = len(data[(data['sex']=='female') & (data['benign_malignant']=='benign')])
    female_with_malignant = len(data[(data['sex']=='female') & (data['benign_malignant']=='malignant')])
    male = [male_with_benign,male_with_malignant]
    female = [female_with_benign,female_with_malignant]

    fig,ax = plt.subplots()
    X = np.arange(2)
    ax.set_title('sex distribution w.r.t benign_malignant ',fontsize=16)
    ax.set_ylabel('counts',fontsize=16)
    rects1 = ax.bar(X-0.125,male,width=0.25,color='g',label='male')
    rects2 = ax.bar(X+0.125,female,width=0.25,color='orange',label='female')
    ax.set_xticks(X)
    ax.set_xticklabels(['benign','malignant'])
    ax.legend((rects1[0],rects2[0]),('male','female'))

    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2.,height,
                    '%d' % int(height),
                    ha='center', va='bottom',fontsize=12)

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('sex distribution_wrt_benign_malignant.jpg')
    plt.show()
    
    
    
def plot_diagnosis_distribution():
    plt.figure(figsize=(10,5))
    ax = data['diagnosis'].value_counts().plot(kind='barh',color='g',fontsize=12)
    ax.set_title('diagnosis distribution',fontsize=16)
    ax.set_ylabel('diagnosis',fontsize=16)
    ax.set_xlabel('counts',fontsize=16)

    for i in ax.patches:
        ax.text(i.get_width(), i.get_y(), str(i.get_width()),fontsize=12)

    plt.tight_layout()
    plt.savefig('diagnosis_distribution.jpg')
    plt.show()
    
    
def plot_location_of_skin_lesion_wrt_gender():
    cancer_sites = data['anatom_site_general_challenge'].unique()
    male = []
    female = []
    for site in cancer_sites:
        no_male = len(data[(data['sex']=='male') & (data['anatom_site_general_challenge']==site)])
        male.append(no_male)
    for site in cancer_sites:
        no_female = len(data[(data['sex']=='female') & (data['anatom_site_general_challenge']==site)])
        female.append(no_female)
    fig,ax = plt.subplots(figsize=(10,6))
    X = np.arange(data['anatom_site_general_challenge'].nunique())
    rects1 = ax.bar(X-0.125,male,width=0.25,color='green',label='male')
    rects2 = ax.bar(X+0.125,female,width=0.25,color='orange',label='female')
    ax.set_title('Location of skin_lesion w.r.t gender',fontsize=16)
    ax.set_ylabel('counts',fontsize=16)
    ax.set_xlabel('skin_lesion site on body',fontsize=16)
    ax.set_xticks(X)
    ax.set_xticklabels(cancer_sites,rotation=25,fontsize=12)
    ax.legend((rects1[0], rects2[0]), ('male', 'female'))


    def autolabel(rects):
        """
        Attach a text label above each bar displaying its height
        """
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2.,height,
                    '%d' % int(height),
                    ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('skin_lesion_site_wrt_gender.jpg')
    plt.show()
    
    
    
def showSomeData(path):
    plt.figure(figsize=(15,5))
    filenames = os.listdir(path)
    for index,filename in enumerate(filenames[:10]):
        file_path = os.path.join(path,filename)
        img = mpimg.imread(file_path)
        plt.subplot(2,5,index+1)
        plt.axis('off')
        plt.imshow(img)
    
    plt.tight_layout()
    plt.savefig(filename)
    
    
def patient_distribution_in_trainAndValidationData():
    print('Total patient in Data : ',data['patient_id'].nunique())
    print('Total patient in Training Data : ',train_data_df['patient_id'].nunique())
    print('Total patient in Validation Data : ',val_data_df['patient_id'].nunique())
    common_patients_df = pd.merge(train_data_df['patient_id'].drop_duplicates(),val_data_df['patient_id'].drop_duplicates(),how='inner')
    print('Common patients in train and valiadtion data : ',common_patients_df.shape[0])
    
    
    

def TrainDataGenerator():
    
    train_datagen = ImageDataGenerator(
                                    rescale = 1./255.,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    shear_range = 0.1, 
                                    zoom_range = 0.1, 
                                    rotation_range=0.1,
                                    horizontal_flip = True,
                                    fill_mode="reflect"
                                    )

    train_batches =  train_datagen.flow_from_dataframe(
    dataframe = train_files_df,
    directory = train_path,
    x_col = 'filename',
    y_col = 'category',
    classes = data['diagnosis'].unique().tolist(),
    target_size = target_size,
    batch_size = batch_size,
    seed=0)
    return train_batches




def ValDataGenerator():
    val_datagen = ImageDataGenerator(
                                 rescale=1./255)

    val_batches = val_datagen.flow_from_dataframe(
    dataframe = val_files_df,
    directory = train_path,
    x_col = 'filename',
    y_col = 'category',
    classes = data['diagnosis'].unique().tolist(),
    target_size = target_size,
    batch_size = batch_size,
    shuffle = False,
    seed = 0)
    return val_batches
    
    
    
def plot_loss():
    plt.plot(log['loss'],color='c',label='loss')
    plt.plot(log['val_loss'],color='y',label='val_loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Loss_Curve.jpg')
    plt.show()
    
    
    
def plot_accuracy():
    plt.plot(log['accuracy'],color='m',label='accuracy')
    plt.plot(log['val_accuracy'],color='b',label='val_accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig('Accuracy_curve.jpg')
    plt.show()
    
    
    
def predict_result(batches):
    model_path = '/home/sudhakar/Project/model_and_log_files/best_model.h5'
    model = load_model(model_path)
    predictions = model.predict(x=batches)
    return predictions
    

def plotImages(imgs,nrows,ncols):
    fig ,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(15,10))
    axes = axes.flatten()
    for img,axis in zip(imgs,axes):
        axis.imshow(img)
        axis.axis('off')
    plt.tight_layout()
    plt.show()



def plot_augmented_images():
    augmented_images = [train_batches[0][0][0] for i in range(10)]
    print('AUGMENTED IMAGES')
    plotImages(augmented_images,nrows=1,ncols=10)
    plt.savefig('augmented_images.jpg')
    
    
    
def build_model():
    model = Sequential([
    Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding = 'same',kernel_initializer='he_uniform',input_shape=(target_size[0],target_size[1],3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same',kernel_initializer='he_uniform'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'same',kernel_initializer='he_uniform'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=256, kernel_size=(3, 3), activation='relu', padding = 'same',kernel_initializer='he_uniform'),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=128,activation='relu'),
    Dense(units=9, activation='softmax')
    ])
    
    
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


def plot_model_architecture():
    plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
    
    
    
def cal_confusion_matrix():
    cm = confusion_matrix(
        y_true = y_true,
        y_pred = y_pred,
        labels = np.arange(9))
    return cm
 
 
 
def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix', cmap=plt.cm.viridis):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title,fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=75,fontsize=14)
    plt.yticks(tick_marks, classes,fontsize=14)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                horizontalalignment="center",
                color="white" if cm[i, j] < thresh else "black",
                ha='center',
                va='bottom',
                fontsize=12)
    
    plt.style.use('default')
    plt.xlabel('Predicted label',fontsize=16)
    plt.ylabel('True label',fontsize=16)
    plt.tight_layout()
    plt.savefig('confusion_matrix.jpg')
 
 
def plot_roc_curve():
    # roc curve for classes
    fpr = {}
    tpr = {}
    thresh ={}

    n_class = 9

    for i in range(n_class):    
        fpr[i], tpr[i], thresh[i] = roc_curve(y_true, y_pred, pos_label=i)
        print(auc(fpr[i],tpr[i]))
        plt.plot(fpr[i],tpr[i],label='Class {i}'.format(i=i))
        
    # plotting    
    plt.title('Multiclass ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='best')
    plt.savefig('roc_curve.jpg')
    plt.show()
    
    
def get_class_weights():
    '''It is needed because in the dataset
    there is a considerable class imbalance'''
    class_weights = class_weight.compute_class_weight('balanced',
                                                 classes=data['diagnosis'].unique(),
                                                 y=data['diagnosis'])
    class_weights = dict(zip(np.arange(9) , class_weights))
    
    return class_weights


def callbacks():
    '''This method returns the list of callbacks'''
    es = EarlyStopping(monitor='val_loss',mode='min',verbose=1,patience=1)
    mc = ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1,patience=2,verbose=1,mode='min',min_lr=0.000001)
    csv_logger = CSVLogger(filename='training.log',separator=',',append=True)
    return [mc,reduce_lr,csv_logger]


    
if __name__=='__main__':
    train_path = '../input/siim-isic-melanoma-classification/jpeg/train/'
    test_path = '../input/siim-isic-melanoma-classification/jpeg/test/'
    
    # data preprocessing
    train_data_df , val_data_df = train_test_split(data,test_size=0.2,random_state=0)
    
    train_files_df = train_data_df[['image_name','diagnosis']]
    val_files_df = val_data_df[['image_name','diagnosis']] 
    
    train_files_df['image_name']=train_files_df['image_name']+'.jpg'
    val_files_df['image_name']=val_files_df['image_name']+'.jpg'
    train_files_df = train_files_df.rename(columns={'image_name':'filename', 'diagnosis':'category'})
    val_files_df = val_files_df.rename(columns={'image_name':'filename', 'diagnosis':'category'})
    
    # train and val batches
    train_batches = TrainDataGenerator()
    val_batches = ValDataGenerator()
    
    
    plot_augmented_images()
    
    
    # model configurations constants
    img_width , img_height, channels = (256,256,3)
    target_size = (img_width , img_height)
    batch_size = 128
    optimizer=Adam
    epochs=12
    verbose=1
    callbacks = callbacks()
    class_weights = get_class_weights()
    
    
    # buid the model
    model = build_model()
    
    # model summary
    model.summary()
    
    plot_model_architecture()
    
    # train the model
    history = model.fit(train_batches,
                    batch_size=batch_size,
                    epochs=20,
                    verbose=1,
                    initial_epoch=0,
                    class_weight=class_weights,
                    validation_data=val_batches,
                    callbacks=callbacks
                    )
    
    # Model evaluation
    
    log = pd.read_csv('./training.log')
    
    # plot loss and accuracy
    plot_loss()
    plot_accuracy()
    
    # Prediction 
    predictions = predict_result(val_batches)
    
    
    y_true = val_batches.classes
    y_pred = np.argmax(predictions, axis=-1)
    
    # Accuracy
    print('Accuracy = ',accuracy_score(val_batches.classes,np.argmax(predictions, axis=-1)))
    
    
    # confusion_matrix
    cm = cal_confusion_matrix()
    print(cm)
    # plot confusion_matrix
    plot_confusion_matrix(cm=cm, classes=c_matrix_labels, title='Confusion Matrix')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
