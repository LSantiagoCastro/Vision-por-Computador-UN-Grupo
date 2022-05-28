import cv2 
import numpy as np
#from matplotlib import matplotlib.pyplot as plt

#from google.colab.patches import cv2_imshow
import cv2
import argparse



class detect_color():

  def __init__(self, img):
    #self.img = cv2.imread(img)
    self.img = img
    self.original = img
    self.img = np.array(self.img, dtype=np.uint8) 
    self.img = np.array(self.img, dtype=np.uint8)
    self.img = cv2.cvtColor(self.img,  cv2.COLOR_BGR2RGB) #Cambia el orden de los canales
    self.imgHsv = cv2.cvtColor(self.img,  cv2.COLOR_BGR2HSV)

  def Corte(self, color):
    if(color == 1):
      verde1 = np.array([40,100,20], np.uint8)
      verde2 = np.array([80,255,255], np.uint8)

      mask1 = cv2.inRange(self.imgHsv, verde1, verde2)
      maskverde = cv2.bitwise_and(self.imgHsv, self.imgHsv, mask = mask1)

      kernel = np.ones((7,7),np.uint8)

      img_hole = mask1 - np.random.randint(2,size=(mask1.shape[0],mask1.shape[1]),dtype=np.uint8)*255
      img_hole_removed = cv2.erode(cv2.dilate(img_hole,kernel),kernel)  

      img_noise = img_hole_removed + np.random.randint(2,size=(img_hole_removed.shape[0],img_hole_removed.shape[1]),dtype=np.uint8)*255
      kernel = np.ones((7,7),np.uint8) 
      mask2 = cv2.dilate(cv2.erode(img_noise,kernel),kernel)
      maskverde = cv2.bitwise_and(self.imgHsv, self.imgHsv, mask = mask2)
      SoloVerde = cv2.cvtColor(maskverde,  cv2.COLOR_HSV2RGB)
      img_color_detected = SoloVerde
      self.verde = SoloVerde

    if(color == 2):
      amarillo1 = np.array([25,100,20], np.uint8)
      amarillo2 = np.array([35,255,255], np.uint8)

      #amarillo1 = np.array([25,103,255], np.uint8)
      #amarillo2 = np.array([35,20,155], np.uint8)

      mask_a_1 = cv2.inRange(self.imgHsv, amarillo1, amarillo2)
      maskamarilla = cv2.bitwise_and(self.imgHsv, self.imgHsv, mask = mask_a_1)

      #kernel = np.ones((7,7),np.uint8)
      kernel = np.ones((7,7),np.uint8)
      img_hole_a = mask_a_1 - np.random.randint(2,size=(mask_a_1.shape[0],mask_a_1.shape[1]),dtype=np.uint8)*255
      img_hole_removed_a = cv2.erode(cv2.dilate(img_hole_a,kernel),kernel)  

      img_noise_a = img_hole_removed_a + np.random.randint(2,size=(img_hole_removed_a.shape[0],img_hole_removed_a.shape[1]),dtype=np.uint8)*255
      kernel = np.ones((7,7),np.uint8) 
      mask_a_2 = cv2.dilate(cv2.erode(img_noise_a,kernel),kernel)
      maskamarilla = cv2.bitwise_and(self.imgHsv, self.imgHsv, mask = mask_a_2)
      Soloamarilla = cv2.cvtColor(maskamarilla,  cv2.COLOR_HSV2RGB)
      img_color_detected = Soloamarilla
      self.amarilla = Soloamarilla


    if(color == 3):

      violeta1 = np.array([145,100,0], np.uint8)
      violeta2 = np.array([170,255,200], np.uint8)

      mask_vio_1 = cv2.inRange(self.imgHsv, violeta1, violeta2)
      maskverde = cv2.bitwise_and(self.imgHsv, self.imgHsv, mask = mask_vio_1)


      img_noise_v = mask_vio_1 + np.random.randint(2,size=(mask_vio_1.shape[0],mask_vio_1.shape[1]),dtype=np.uint8)*255

      kernel = np.ones((7,7),np.uint8) 

      img_noise_removed_v = cv2.dilate(cv2.erode(img_noise_v,kernel),kernel)


      img_hole_v = img_noise_removed_v - np.random.randint(2,size=(img_noise_removed_v.shape[0],img_noise_removed_v.shape[1]),dtype=np.uint8)*255


      img_hole_removed_v = cv2.erode(cv2.dilate(img_hole_v,kernel),kernel) 

      img_noise_v = img_hole_removed_v + np.random.randint(2,size=(img_hole_removed_v.shape[0],img_hole_removed_v.shape[1]),dtype=np.uint8)*255
      kernel = np.ones((7,7),np.uint8) 
      mask_vio_2 = cv2.dilate(cv2.erode(img_noise_v,kernel),kernel)
      maskvioleta = cv2.bitwise_and(self.imgHsv, self.imgHsv, mask = mask_vio_2)
      SoloVioleta = cv2.cvtColor(maskvioleta,  cv2.COLOR_HSV2RGB)
      img_color_detected = SoloVioleta
      self.morado = SoloVioleta
    return img_color_detected

  
  def Conteo_total(self):
     #Deteccion de color VERDE ###############################################
      verde_deteccion = self.Corte(1)
      verde_conteo = verde_deteccion

      #Transformacion del espacio de color bgr a gray y extraccion de bordes
      grises_verdes = cv2.cvtColor(verde_deteccion, cv2.COLOR_BGR2GRAY)
      bordes_verdes =  cv2.Canny(grises_verdes, 10, 631)

      ctns, _ = cv2.findContours(bordes_verdes, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      #cv2.drawContours(verde, ctns, -1, (0,0,255), 2)
      color = (255,255,255)
      #print('Número de contornos encontrados: ', len(ctns))
      texto = 'Verdes: '+ str(len(ctns))
      cv2.putText(verde_conteo, texto, (10,50), cv2.FONT_ITALIC, 2,
        color, 2, cv2.LINE_AA)

      #Deteccion de color AMARILLO#############################################
      amarillo_deteccion = self.Corte(2)
      amarillo_conteo = amarillo_deteccion

      #Transformacion del espacio de color bgr a gray y extraccion de bordes
      grises_amarillo = cv2.cvtColor(amarillo_deteccion, cv2.COLOR_BGR2GRAY)
      bordes_amarillo =  cv2.Canny(grises_amarillo, 10, 420)

      ctns2, _ = cv2.findContours(bordes_amarillo, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      #cv2.drawContours(verde, ctns, -1, (0,0,255), 2)
      color = (255,255,255)
      #print('Número de contornos encontrados: ', len(ctns))
      texto = 'Amarillos: '+ str(len(ctns2))
      cv2.putText(amarillo_conteo, texto, (10,50), cv2.FONT_ITALIC, 2,
        color, 2, cv2.LINE_AA)

      #Deteccion de color VIOLETA###############################################
      violeta_deteccion = self.Corte(3)
      violeta_conteo = violeta_deteccion

      #Transformacion del espacio de color bgr a gray y extraccion de bordes
      grises_violeta = cv2.cvtColor(violeta_deteccion, cv2.COLOR_BGR2GRAY)
      bordes_violeta =  cv2.Canny(grises_violeta, 0, 150)

      ctns3, _ = cv2.findContours(bordes_violeta, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
      #cv2.drawContours(verde, ctns, -1, (0,0,255), 2)
      color = (255,255,255)
      #print('Número de contornos encontrados: ', len(ctns))
      texto = 'Violetas: '+ str(len(ctns3))
      cv2.putText(violeta_conteo, texto, (10,50), cv2.FONT_ITALIC, 2,
        color, 2, cv2.LINE_AA)
      
      verde_con = cv2.cvtColor(verde_conteo,  cv2.COLOR_RGB2BGR) 
      amarillo_con = cv2.cvtColor(amarillo_conteo,  cv2.COLOR_RGB2BGR)
      violeta_con = cv2.cvtColor(violeta_conteo,  cv2.COLOR_RGB2BGR)

  


      imgs = np.hstack([ self.original ,verde_con, amarillo_con, violeta_con])
      imgs = cv2.resize(imgs, (300*4, 200))

      
      # Mostrar múltiples
      cv2.imshow("main", imgs)
     


