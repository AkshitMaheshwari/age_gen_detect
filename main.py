# import cv2
# import cv2.dnn
# import numpy as np
# from keras.models import model_from_json
# import streamlit as st
#
# GENDER_MODEL = 'deploy_gender.prototxt'
# GENDER_PROTO = 'gender_net.caffemodel'
# MODEL_MEAN_VALUES = (78.4263377603,87.76891443744,114.89584746)
# GENDER_LIST = ["Male","Female"]
#
# FACE_PROTO = "deploy.prototxt.txt"
# FACE_MODEL = "res10_300*300_ssd_iter_140000_fp16.caffemodel"
#
# AGE_MODEL = 'deploy_age.prototxt'
# AGE_PROTO = 'age_net.caffemodel'
# AGE_INTERVALS = ['(0,2)','(4,6)','(8,12)','(15,20)','(25,32)','(38,43)','(48,53)','(60,100)']
#
# with open('emotion_model.json','r') as json_file:
#     loaded_model_json = json_file.read()
# emotion_model = model_from_json(loaded_model_json)
#
# emotion_model.load_weights('emotion_model_weights.h5')
# emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]
#
# frame_width = 1280
# frame_height = 720
#
# face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO,FACE_MODEL)
# age_net = cv2.dnn.readNetFromCaffe(AGE_MODEL,AGE_PROTO)
# gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL,GENDER_PROTO)
#
# def get_faces(frame,confidence_threshold=0.5):
#     blob = cv2.dnn.blobFromImage(frame,1.0,(300,300),(104,177.0,123.0))
#     face_net.setInput(blob)
#     output = np.squeeze(face_net.forward())
#     faces = []
#
#     for i in range(output.shape[0]):
#         confidence = output[i,2]
#         if confidence>confidence_threshold:
#             box = output[i,3:7]*np.array([frame.shape[1],frame.shape[0],frame.shape[1],frame.shape[0]])
#             start_x,start_y,end_x,end_y = box.astype(int)
#             start_x, start_y, end_x, end_y = start_x -10,start_y-10,end_x+10,end_y+10
#             start_x=0 if start_x<0 else start_x
#             start_y = 0 if start_y < 0 else start_y
#             end_x = 0 if end_x < 0 else end_x
#             end_y = 0 if end_y < 0 else end_y
#             faces.append((start_x,start_y,end_x,end_y))
#     return faces
#
# def image_resize(image,width=None,height=None,inter=cv2.INTER_AREA):
#     dim = None
#     (h,w) = image.shape[:2]
#     if width is None and height is None:
#         return image
#     if width is None:
#         r = height/float(h)
#         dim = (int(w*r),height)
#     else:
#         r = width/float(w)
#         dim = (width,int(h*r))
#     return cv2.resize(image,dim,interpolation=inter)
#
# def get_gender_predictions(face_img):
#     blob = cv2.dnn.blobFromImage(image=face_img,scalefactor=1.0,size=(227,227),mean = MODEL_MEAN_VALUES,swapRB=False,crop=False)
#     gender_net.setInput(blob)
#     return gender_net.forward()
#
# def age_predictions(face_img):
#     blob = cv2.dnn.blobFromImage(image=face_img, scalefactor=1.0, size=(227, 227), mean=MODEL_MEAN_VALUES, swapRB=False)
#     age_net.setInput(blob)
#     return age_net.forward()
#
# def get_emotion_predictions(face_img):
#     face_img = cv2.cvtColor(face_img,cv2.COLOR_BGR2GRAY)
#     face_img = cv2.resize(face_img,(48,48))
#     face_img = np.expand_dims(face_img,axis=0)
#     face_img = face_img/255.0
#
#     emotion_preds = emotion_model.predict(face_img)
#     return emotion_preds
#
# def predict_age_gender_and_emotion():
#     stframe = st.empty()
#     cap = cv2.VideoCapture(0)
#     quit_button = st.button("Quit")
#     unique_key = 0
#
#     while not quit_button:
#         _, img = cap.read()
#         frame = img.copy()
#
#         if frame.shape[1] > frame_width:
#             frame = image_resize(frame,width= frame_width)
#         faces = get_faces(frame)
#         for i, (start_x,start_y,end_x,end_y) in enumerate(faces):
#             face_img = frame[start_y:end_y,start_x:end_x]
#
#             age_preds = age_predictions(face_img)
#             gender_preds = get_gender_predictions(face_img)
#             emotion_preds = get_emotion_predictions(face_img)
#
#             i = emotion_preds[0].argmax()
#             emotion = emotion_labels[i]
#             emotion_confidence_score = emotion_preds[0][i]
#
#             i = gender_preds[0].argmax()
#             gender = GENDER_LIST[i]
#             gender_confidence_score = gender_preds[0][i]
#
#             i = age_preds[0].argmax()
#             age = AGE_INTERVALS[i]
#             age_confidence_score = age_preds[0][i]
#
#             label = f"{gender}, {age}, {emotion}-{emotion_confidence_score*100:.1f}%"
#             ypos = start_y-15
#             while ypos < 15:
#                 ypos +=15
#             box_color =(255,0,0) if gender == "Male" else (147,20,255)
#             cv2.rectangle(frame,(start_x,start_y),(end_x,end_y),box_color,2)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             font_scale = 0.54
#
#             text_x = start_x
#             text_y = ypos-5
#
#             cv2.putText(frame,f"{gender}, {age}",(text_x,text_y),font, font_scale,box_color,2)
#             text_y_emotion = text_y-20
#             cv2.putText(frame,f"Emotion: {emotion}-{emotion_confidence_score*100:.1f}%",(text_x,text_y_emotion),font,font_scale,box_color,2)
#         stframe.image(frame,channels='BGR')
#     cap.release()
# def main():
#     predict_age_gender_and_emotion()
# if __name__ == "__main__":
#     main()
#
# # Press the green button in the gutter to run the script.
#
#
# # See PyCharm help at https://www.jetbrains.com/help/pycharm/

import cv2
import streamlit as st
import numpy as np

# Function to highlight faces
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

# Load models
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Streamlit app setup
st.title("Real-Time Age and Gender Detection")
run = st.checkbox('Run Detection')
quit_app = st.button('Quit')

# Capture video using OpenCV
if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        if not faceBoxes:
            st.warning("No face detected")

        for faceBox in faceBoxes:
            face = frame[max(0, faceBox[1] - 20):min(faceBox[3] + 20, frame.shape[0] - 1),
                         max(0, faceBox[0] - 20):min(faceBox[2] + 20, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]

            cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the frame using Streamlit
        stframe.image(cv2.cvtColor(resultImg, cv2.COLOR_BGR2RGB), channels="RGB")

        # Stop running if the Quit button is pressed
        if quit_app:
            cap.release()
            break

    cap.release()
st.write("Detection stopped.")

