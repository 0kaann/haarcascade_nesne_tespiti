# opencv
import cv2



# numpy
import numpy as np
import matplotlib.pyplot as plt



# resmi siyah-beyaz olarak içe aktardık
img=cv2.imread("odev2.jpg")
gray_img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.figure(),plt.imshow(gray_img,cmap="gray"),plt.axis("off")
color=(0,255,0)



# resimde ki kenarları tespit ettik
edges=cv2.Canny(image=img,threshold1=85,threshold2=255)
plt.figure(),plt.imshow(edges,cmap="gray"),plt.axis("off")


#threshold için medyan değeri bulduk genelde en iyi değerdir
med_val=np.median(img)



# low,high değerlerini belirledik
low=int(max(0,(1-0.33)*med_val))
high=int(max(255,(1+0.33)*med_val))



#kenar tespiti 
update_edges=cv2.Canny(image=img, threshold1=low, threshold2=high)
plt.figure(),plt.imshow(update_edges,cmap="gray"),plt.axis("off")



#resimde ki gürültüyü bulanıklık yöntemi ile azaltmaya çalıştık
blurred_img=cv2.blur(update_edges, ksize=(5,5))



#medyan değeri belirledik
med_val2=np.median(blurred_img)



# low,high değerlerini belirledik
low_=int(max(0,(1-0.33)*med_val2))
high_=int(max(255,(1+0.33)*med_val2))



#kenar tespiti 
update_edges2=cv2.Canny(image=blurred_img, threshold1=low_, threshold2=high_)
plt.figure(),plt.imshow(update_edges2,cmap="gray"),plt.axis("off")



#yüz tespiti için haar cascade'i içe aktarıp kullanıyoruz
cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_rects=cascade.detectMultiScale(img)



#tespit edilen kordinatları görselleştiriyoruz
for (i,(x,y,w,h)) in enumerate(face_rects):
    cv2.rectangle(img, (x,y),(x+w,y+h), color)
    cv2.putText(img, ("{}.yuz".format(i+1)), (x,y-7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color)
cv2.imshow("ayar",img)



# insan tespiti algoritmasını çağırıyoruz
hog=cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
(rects,weights)=hog.detectMultiScale(img,padding=(2,2),scale=1.05)



#tespit edilen kordinatları görselleştiriyoruz
for (x,y,w,h) in rects:
    cv2.rectangle(img, (x,y),(x+w,y+h), color)
plt.figure(),plt.imshow(img,cmap="gray"),plt.axis("off")
    















