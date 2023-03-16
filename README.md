# 📌 Facial-Expressions-Recognition
Deep Learning-based Flask web app that predicts the Facial Expressions of users in real time, by image or image url.

I have trained a CNN model for facial expression detection with the FER2013 dataset containing 7 emotion classes
('Angry'😡 , 'Disgust'😖, 'Fear'😥, 'Happy'😄, 'Sad'😢, 'Surprise'😲, 'Neutral'🙂).

Using flask framework to build a web application.


## ◉ In this project the Model can take input through following ways :

##### 1- Real-time Video input 

##### 2- Upload Images from your pc  

##### 3- Enter image URL 

👇🏻
![start](https://user-images.githubusercontent.com/74102314/225615927-28821c8f-c790-4345-ad30-0f331f88bb6f.PNG)
 
### 1-  Real-time Video input 👇🏻
![real2](https://user-images.githubusercontent.com/74102314/225615449-a0c0cad8-2e3d-4deb-a4aa-27e5a212522d.PNG)

### 2-  Upload Images from your pc 👇🏻
![up_img](https://user-images.githubusercontent.com/74102314/225615592-41246dd8-b102-433a-9b59-650148ef57ce.PNG)
 
### 3-  Enter URL of the Image 👇🏻
![up_url](https://user-images.githubusercontent.com/74102314/225615720-a6cbcd3c-0311-4557-a70b-d8a21bf22817.PNG)
### ■   Prediction 👇🏻
![pred_url](https://user-images.githubusercontent.com/74102314/225615689-58d41f76-7e88-41a6-b33a-00867bba2e6a.PNG)
### ■   NO_Prediction 👇🏻
![no_pred](https://user-images.githubusercontent.com/74102314/225615861-ad6fe815-a0ba-4c35-b6d8-82b61598d750.PNG)

◉ This project involves several technologies and tools , Here are some commonly used ones:

• Python: the primary programming language used in this project to implement the CNN model, the Flask web application, and the data processing pipeline.

• TensorFlow :deep learning framework that is used to implement the CNN model for facial expression recognition.

• OpenCV: open-source computer vision library that is used for image and video processing ,capture live video feeds,and perform other image processing tasks.

• Flask: micro web framework that is used to build the web application that runs the facial expression recognition model.

• HTML/CSS/JavaScript: for building the web front-end. 


## ◉ The objectives of the project are as follows: 

• Import the required Packages and Libraries. 

• explore the dataset FER-2013.

• Data analysis,Data Augmentation and Creating Training and Validation Batches.  

• Create a CNN using 4 Convolutional Layers including Batch Normalization, Activation, Max Pooling, Dropout Layers followed by Flatten Layer, 2 Fully Connected dense Layers and finally Dense Layer with SoftMax Activation Function.    

• Compile the model using Adam Optimizer and categorical cross entropy loss function.   

• Training the model for 15 epochs and then Evaluating the model as well as saving the model Weights in .h5 Value.  

• Saving the model as JSON string.   

• create a Flask app to serve predictions. 

• design an HTML templates for the Flask app.  

• Finally we can use the model to recognize facial expressions by real time video ,from image or by image url.    
 


