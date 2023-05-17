from utils import load_data
from flask import Flask,request,render_template
from PIL import Image
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split

app=Flask(__name__)
(feature,labels)=load_data()

x_train , x_test , y_train , y_test=train_test_split(feature,labels,test_size=0.1)

categories=['daisy','dandelion','kale'];

num_classes=len(categories);

input_layer=tf.keras.layers.Input([224,224,3])

conv1=tf.keras.layers.Conv2D(filters=32 , kernel_size=(5,5),padding='Same',activation='relu')(input_layer)
pool1=tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)

conv2=tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding='Same',activation='relu')(pool1)
pool2=tf.keras.layers.MaxPooling2D(pool_size=(3,3),strides=(2,2))(conv2)

conv3=tf.keras.layers.Conv2D(filters=96 , kernel_size=(3,3) , padding='Same',activation='relu')(pool2)
pool3=tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv3)


conv4=tf.keras.layers.Conv2D(filters=96,kernel_size=(3,3) , padding='Same',activation='relu')(pool3)
pool4=tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv4)

#conv5=tf.keras.layers.Conv2D(filters=96,kernel_size=(3,3) , padding='Same',activation='relu')(pool4)
#pool5=tf.keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv5)



flt1=tf.keras.layers.Flatten()(pool4);

dn1=tf.keras.layers.Dense(512,activation='relu')(flt1)#rectifying linear Unit
out=tf.keras.layers.Dense(num_classes,activation='softmax')(dn1)



model=tf.keras.Model(input_layer,out)

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=10,epochs=5)

model.save('Identification Of Plants.h5')



plant_names = {
    0:'daisy',
    1:'dandelion',
    2:'kale'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        img_array = np.expand_dims(np.array(Image.open(image).resize((224, 224))), axis=0)
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        plant_name = plant_names[predicted_label]
        confidence = prediction[0][predicted_label] * 100

        return render_template('result.html', plant_name=plant_name, confidence=confidence);
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
