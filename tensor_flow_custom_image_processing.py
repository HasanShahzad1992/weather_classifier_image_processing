import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import models,layers
import joblib

all_images=[]
all_labels=[]
sub_folders=os.listdir("dataset")
for i in sub_folders:
    sub_folder_path=os.path.join("dataset",i)
    opening_sub_folder=os.listdir(sub_folder_path)
    for j in opening_sub_folder:
        image_path=os.path.join("dataset",i,j)
        opening_images=image.load_img(image_path,target_size=(128,128))
        image_array=image.img_to_array(opening_images)
        normalized_images=image_array/255
        all_images.append(normalized_images)
        all_labels.append(i)
print(all_images)
print(all_labels)
# the all_images and all_labels need to be converted to numpy
numpy_array_images=np.array(all_images)
numpy_array_labels=np.array(all_labels)
#we create label_encoder variable which uses LabelEncoder() function
#this label_encoder uses fit_transform to get a data_set like [0,0,0,1,1,1,2,2,3]
#this label_encoder has a function of .classes_ but it can only be used after fit_transform and it tell the column names of dataset
label_encoder=LabelEncoder()
encoded_labels=label_encoder.fit_transform(numpy_array_labels)

#we have done labelling, now we need to make one hot encoding table
#encoded labels is a dataset of [0,0,0,1,1,1,] and we give length of classes to tell the number of  unique column names would be there
length_of_classes=len(label_encoder.classes_)
table_one_hot_encoded_labels=to_categorical(encoded_labels,length_of_classes)

X=numpy_array_images
Y=table_one_hot_encoded_labels

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
#for X_train, we get shape(5489, 128, 128, 3) which means it has 5489 images, its has height and width of 128 and 128, and 3 colour coded
#for Y_train, we get shape(5489,11), 5489 is the labels of 5489 images and 11 is the total column number of categories


print(X_train.shape)
print(X_test.shape)
model=models.Sequential([layers.Conv2D(32,(3,3),activation="relu",input_shape=(128,128,3)),layers.MaxPool2D((2,2)),layers.Conv2D(64,(3,3),activation="relu"),layers.MaxPool2D((2,2)),layers.Conv2D(128,(3,3),activation="relu"),layers.Flatten(),layers.Dense(128,activation="relu"),layers.Dense(length_of_classes,activation="softmax")])
#sparsecategoricalcrossentropy, sparse is only used when onehotencoded table is not made, otherwise if one hot encoded table is made, so only categoricalCrossentropy, not sparse
#sparse is representation of labels as just integers like 1,2,3, representation of labels as binary like 1 or 0, then it is not sparse
model.compile(optimizer="adam",loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=20)
model.save("weather_classifier.keras")
test_loss,test_accuracy=model.evaluate(X_test,Y_test,verbose=2)

loaded_model=models.load_model("weather_classifier.keras")
image_load=image.load_img("weather_image.jpg",target_size=(128,128))
image_array=image.img_to_array(image_load)
expand_dimenstions=np.expand_dims(image_array,axis=0)
normalized_image_array=expand_dimenstions/255
prediction=loaded_model.predict(normalized_image_array)
predict_digit=np.argmax(prediction)
label_encoder=joblib.load("label_encoder.joblib")
#in label_encoder, it only saves labels like summer, winter, spring,unique names.,and .classes_ gives a list
#after we load label encoder, there may be other garbage value that is not required so thats why we do label_encoder.classes_
predicted_label=label_encoder.classes_[predict_digit]
print(predicted_label)
