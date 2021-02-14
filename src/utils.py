about_dataset = '''
    The dataset was generated by the International Skin Imaging Collaboration (ISIC)
    and images are from the following sources: \n
    ⭕ Hospital Clínic de Barcelona\n
    ⭕ Medical University of Vienna, Memorial Sloan Kettering Cancer Center\n
    ⭕ Melanoma Institute Australia\n
    ⭕ The University of Queensland, and the\n
    ⭕ University of Athens Medical School.

    ### Columns
    - image_name - unique identifier
    - patient_id - unique patient identifier
    - sex        - the sex of the patient (when unknown, will be blank)
    - age_approx - approximate patient age at time of imaging
    - anatom_site_general_challenge - location of imaged site
    - diagnosis  - detailed diagnosis information (train only)
    - benign_malignant - indicator of malignancy of imaged lesion
    - target - binarized version of the target variable
    '''

what_will_it_predict = '''
    ## 2️⃣ What will it predict ?
    It will predict a binary target for each image. This model will predict the probability (floating point) between 0.0 and 1.0 that the lesion in the image is malignant (the target). In the training data, train.csv, the value 0 denotes benign, and 1 indicates malignant.

    '''

data_augmentation_code = '''

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

'''
    
    
    
menu_contents = {'about_dataset':about_dataset , 'what_will_it_predict':what_will_it_predict , 'data_augmentation_code':data_augmentation_code}

