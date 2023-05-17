from utils import load_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import train_test_split

(feature,labels)=load_data()

x_train , x_test , y_train , y_test=train_test_split(feature,labels,test_size=0.1)

categories=['daisy','dandelion','kale']

model=tf.keras.models.load_model('Identification Of Plants.h5')



prediction=model.predict(x_test)

plt.figure(figsize=(9,9))

for i in range((9)):
    plt.subplot(3,3,i+1)
    plt.imshow(x_test[i])
    plt.xlabel('Acutual : ' +categories[y_test[i]]+'\n'+'Predicted :'+categories[np.argmax(prediction[i])])
    print('\n');

    plt.xticks([])
plt.show()