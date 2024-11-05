import cv2
import qrcode
filename="./qr_2.jpg"
content="朱然"
img=qrcode.make(content).save(filename)
x=cv2.QRCodeDetector()
val,_,_=x.detectAndDecode(cv2.imread(filename))
print(val)