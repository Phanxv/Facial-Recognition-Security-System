import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
import cv2, queue, threading, time
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

arduino = serial.Serial(port='/dev/ttyACM0', baudrate=115200, timeout=.1) #<-- edit port

def servo_trigger(frame_w, x_pos, face_w) :
    mid_point = x_pos + face_w/2
    if mid_point < frame_w/2 - face_w :
        #trigger turn left event placeholder (I'm too lazy rn.)
        payload = "LEFT"
        arduino.write(payload.encode())
        print('LEFT')
    elif mid_point > frame_w/2 + face_w :
        payload = "RIGHT"
        arduino.write(payload.encode())
        print('RIGHT')
    else :
       #trigger stop event placeholder 
       payload = "MID"
       arduino.write(payload.encode())
       print('MID')

def main():
    model_path = 'model/third_model_face.tf'
    print("\n [INFO] Loading Model")
    model = load_model(model_path)
    print("\n [INFO] Load Model complete")

    label = ['bus','elf','petch','unknown']

    cap = VideoCapture(1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

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
        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            predictions = model.predict(image_preprocess(img[y:y+h,x:x+w]), verbose=0)
            id = predictions[0].argmax()
            confidence = predictions[0][predictions[0].argmax()] * 100
            if (confidence > 75):
                id = label[id]
                cv2.putText(img, str(id), (x+5,y-5), font, 1, (0,255,0), 2)
                cv2.putText(img, str(confidence), (x+5,y+h-5), font, 0.5, (0,255,0), 1)
                cv2.putText(img, str(f'x:{x} y:{y}'), (x+5,y+h+20), font, 0.5, (0,255,0), 1)
                cv2.putText(img, str(f'w:{w} h:{h}'), (x+5,y+h+35), font, 0.5, (0,255,0), 1)   
            else:
                id = "UNKNOWN"
                cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,0,0), 2)
                cv2.putText(img, str(f'x:{x} y:{y}'), (x+5,y+h+20), font, 0.5, (255,0,0), 1)
                cv2.putText(img, str(f'w:{w} h:{h}'), (x+5,y+h+35), font, 0.5, (0,255,0), 1)  
            servo_trigger(frame_w=frame_width, x_pos=x, face_w=w)   
            
        cv2.imshow('camera',img)
        k = cv2.waitKey(10) & 0xff
        if k == 27:
            break
    print("\n [INFO] Exiting Program and cleanup stuff")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__" :
   main()