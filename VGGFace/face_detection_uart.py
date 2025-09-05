import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2, queue, threading, time
import os
import serial

class VideoCapture:

  def __init__(self, name):
    self.cap = cv2.VideoCapture(name)
    self.q = queue.Queue()
    t = threading.Thread(target=self._reader)
    t.daemon = True
    t.start()

  def _reader(self):
    while True:
      ret, frame = self.cap.read()
      if not ret:
        break
      if not self.q.empty():
        try:
          self.q.get_nowait()
        except queue.Empty:
          pass
      self.q.put(frame)

  def read(self):
    return self.q.get()
  
  def release(self):
    self.cap.release()
  
  def get(self, cv2_macro):
    self.cap.get(cv2_macro)

def image_preprocess(sample) :
    sample_reshape = cv2.resize(sample,dsize=(224,224))
    sample = tf.convert_to_tensor(sample_reshape, dtype=tf.float32)
    sample = tf.expand_dims(sample, 0)
    return sample

rx65n = serial.Serial(port='COM10', baudrate=115200, timeout=.1) #<-- edit port

def send_result_to_rx65n(payload) :
    rx65n.write(payload.encode())
       
def main():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    model_path = f'{dir_path}/model/model_2025-09-04_1756974610.074763.tf'
    print("\n [INFO] Loading Model")
    model = load_model(model_path)
    print("\n [INFO] Load Model complete")

    label = ['aonan','elf','unknown']

    cap = VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier(f'{dir_path}/haarcascades/haarcascade_frontalface_default.xml')

    while(True):
        img = cap.read()
        frame_width = img.shape[1]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(10, 10)
        )
        echo = rx65n.read()
        print(echo )
        if len(faces) > 0 :
          for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            predictions = model.predict(image_preprocess(img[y:y+h,x:x+w]), verbose=0)
            id = predictions[0].argmax()
            confidence = predictions[0][predictions[0].argmax()] * 100
            if (confidence > 90):
                payload = str(int(id))
                send_result_to_rx65n(payload)
                name = label[id]
                cv2.putText(img, str(name), (x+5,y-5), font, 1, (0,255,0), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 0.5, (0,255,0), 1)
                cv2.putText(img, str(f'x:{x} y:{y}'), (x+5,y+h+20), font, 0.5, (0,255,0), 1)
                cv2.putText(img, str(f'w:{w} h:{h}'), (x+5,y+h+35), font, 0.5, (0,255,0), 1)   
            else:
                id = "UNKNOWN"
                payload = str(int(2))
                send_result_to_rx65n(payload)
                cv2.putText(img, str(id), (x+5,y-5), font, 1, (0,0,255), 2)
                cv2.putText(img, str(f'x:{x} y:{y}'), (x+5,y+h+20), font, 0.5, (0,0,255), 1)
                cv2.putText(img, str(f'w:{w} h:{h}'), (x+5,y+h+35), font, 0.5, (0,0,255), 1) 
        else:
          payload = 'x'
          send_result_to_rx65n(payload)
        cv2.imshow('camera',img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    print("\n [INFO] Exiting Program and cleanup stuff")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__" :
   main()