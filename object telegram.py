import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
import dataset
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
import numpy as np
import os
import cv2
import time
import serial
import sys
import sys
import dataset
import telepot
import urllib.request
start = time.time()
video = cv2.VideoCapture(0)
time.sleep(2)
kk=0
tr1=19
ec1=26
GPIO.setup(ec1,GPIO.IN)
GPIO.setup(tr1,GPIO.OUT)
#CATEGORIES = ["Un Riped","Riped", "Damaged"]
CATEGORIES = ["Potholes","Roads"]
print(CATEGORIES[0])
print(CATEGORIES[1])
#print(CATEGORIES[2])
def handle(msg):
  global telegramText
  global chat_id
  global receiveTelegramMessage
  
  chat_id = msg['chat']['id']
  telegramText = msg['text']
  
  print("Message received from " + str(chat_id))
  
  if telegramText == "/start":
    bot.sendMessage(chat_id, "Welcome to ROBOT Bot")
  
  else:
    buz.beep(0.1, 0.1, 1)
    receiveTelegramMessage = True
def capture():
    
    print("Sending photo to " + str(chat_id))
    bot.sendPhoto(chat_id, photo = open('./image.jpg', 'rb'))


bot = telepot.Bot('add ur bot id')
chat_id='6278125319'
bot.message_loop(handle)

print("Telegram bot is ready")

bot.sendMessage(chat_id, 'BOT STARTED')
def GPS_Info():
    global NMEA_buff
    global lat_in_degrees
    global long_in_degrees
    nmea_time = []
    nmea_latitude = []
    nmea_longitude = []
    nmea_time = NMEA_buff[0]                    #extract time from GPGGA string
    nmea_latitude = NMEA_buff[1]                #extract latitude from GPGGA string
    nmea_longitude = NMEA_buff[3]               #extract longitude from GPGGA string
    
    #print("NMEA Time: ", nmea_time,'\n')
    #print ("NMEA Latitude:", nmea_latitude,"NMEA Longitude:", nmea_longitude,'\n')
    try:
        lat = float(nmea_latitude)                  #convert string into float for calculation
        longi = float(nmea_longitude)               #convertr string into float for calculation
    except:
        lat=0
        longi=0
    lat_in_degrees = convert_to_degrees(lat)    #get latitude in degree decimal format
    long_in_degrees = convert_to_degrees(longi) #get longitude in degree decimal format

def convert_to_degrees(raw_value):
    decimal_value = raw_value/100.00
    degrees = int(decimal_value)
    mm_mmmm = (decimal_value - int(decimal_value))/0.6
    position = degrees + mm_mmmm
    position = "%.4f" %(position)
    return position
gpgga_info = "$GPGGA,"
ser = serial.Serial ("/dev/ttyUSB0",timeout=1)              #Open port with baud rate
GPGGA_buffer = 0
NMEA_buff = 0
lat_in_degrees = 0
long_in_degrees = 0
cnt=0

if(1):

      GPIO.setwarnings(False)
      GPIO.setmode(GPIO.BCM)       # Use BCM GPIO numbers
       # switch
      while True:
            GPIO.output(tr1, True)
            time.sleep(0.00001)
            GPIO.output(tr1, False)

            while GPIO.input(ec1)==0:
                  pulse_start = time.time()

            while GPIO.input(ec1)==1:
                  pulse_end = time.time()

            pulse_duration = pulse_end - pulse_start
            distance = pulse_duration * 17150
            d1 = round(distance+1.15, 2)
            time.sleep(1)
            print ("DISTANCE-1           :" + str(d1))
       

            (grabbed, frame) = video.read()
            if not grabbed:
               break
            cv2.imshow("input", frame)
            cv2.waitKey(1)
            cnt=cnt+1
            print(cnt)
            if(d1<30):
              cnt=0
              cv2.imwrite('./data/test/test.jpg',frame)
              cv2.waitKey(1)
##        cv2.imwrite('image.jpg',frame)
##        capture()
##        time.sleep(5)
##        bot.sendMessage(chat_id, "VIGNAN LARA")
    
    # Path of  training images
              train_path = './data/train'
              if not os.path.exists(train_path):
                 print("No such directory")
                 raise Exception
        # Path of testing images
              dir_path = './data/test'
              if not os.path.exists(dir_path):
                 print("No such directory")
                 raise Exception
        
        # Walk though all testing images one by one
              for root, dirs, files in os.walk(dir_path):
                 for name in files:

                    print("")
                    image_path = name
                    filename = dir_path +'/' +image_path
                    print(filename)
                    image_size=128
                    num_channels=3
                    images = []
            
                    if os.path.exists(filename):
                    
                    # Reading the image using OpenCV
                      image = cv2.imread(filename)
                    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
                      image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
                      images.append(image)
                      images = np.array(images, dtype=np.uint8)
                      images = images.astype('float32')
                      images = np.multiply(images,1.0/255.0) 
                
                    # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
                      x_batch = images.reshape(1, image_size,image_size,num_channels)

                    # Let us restore the saved model 
                      sess = tf.Session()
                    # Step-1: Recreate the network graph. At this step only graph is created.
                      saver = tf.train.import_meta_graph('model/trained_model.meta')
                    # Step-2: Now let's load the weights saved using the restore method.
                      saver.restore(sess, tf.train.latest_checkpoint('./model/'))

                    # Accessing the default graph which we have restored
                      graph = tf.get_default_graph()

                    # Now, let's get hold of the op that we can be processed to get the output.
                    # In the original network y_pred is the tensor that is the prediction of the network
                      y_pred = graph.get_tensor_by_name("y_pred:0")

                    ## Let's feed the images to the input placeholders
                      x= graph.get_tensor_by_name("x:0") 
                      y_true = graph.get_tensor_by_name("y_true:0") 
                      y_test_images = np.zeros((1,len(os.listdir(train_path)))) 


                    # Creating the feed_dict that is required to be fed to calculate y_pred 
                      feed_dict_testing = {x: x_batch, y_true: y_test_images}
                      result=sess.run(y_pred, feed_dict=feed_dict_testing)
                    # Result is of this format [[probabiliy_of_classA probability_of_classB ....]]
                      print(result)

                    # Convert np.array to list
                    a = result[0].tolist()
                    r=0

                    # Finding the maximum of all outputs
                    max1 = max(a)
                    index1 = a.index(max1)
                    predicted_class = None
                    print('INDEX:'+str(index1))
                    predicted_class = CATEGORIES[index1] + " Conf:"+str((result[0][index1])*100)
                    pred=predicted_class
                    if(CATEGORIES[index1]=='Potholes'):
                       print('sending Alert')
                       bot.sendMessage(chat_id,'Pothole detected at')
                       bot.sendMessage(chat_id,'http://maps.google.com/?q=' + str(16.4963) + ',' + str(80.5007) )
                   # send_sms()
                    if(CATEGORIES[index1]=='Roads'):
                       print('Roads are detected')
                    # Walk through directory to find the label of the predicted output
                    count = 0
                    for root, dirs, files in os.walk(train_path):
                        for name in dirs:
                            if count==index1:
                                predicted_class = name
                            count+=1

                    # If the maximum confidence output is largest of all by a big margin then
                    # print the class or else print a warning
                    for i in a:
                        if i!=max1:
                            if max1-i<i:
                                r=1                           
                    if r ==0:
                        print(predicted_class)
                    else:
                        print("Could not classify with definite confidence")
                        print("Maybe:",predicted_class)

                # If file does not exist
         #   else:
        #              print("File does not exist")
