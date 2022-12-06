import cv2
import pytesseract
import webbrowser
 

pytesseract.pytesseract.tesseract_cmd = "/Users/abno2018/depthai/ass4/p_2_abner/Tesseract-OCR//tesseract.exe"

img = cv2.imread("abnercard.png")
 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
 
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
 
dilat = cv2.dilate(thresh, rect_kernel, iterations = 1)
 
contours, hierarchy = cv2.findContours(dilat, cv2.RETR_EXTERNAL,
                                                 cv2.CHAIN_APPROX_NONE)
 
img2 = img.copy()
 
file = open("business_card_info.txt", "w+")
file.write("")
file.close()

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    rect = cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cropped = img2[y:y + h, x:x + w]

    file = open("business_card_info.txt", "a")

    text = pytesseract.image_to_string(cropped)

    file.write(text)
    file.close

detect = cv2.QRCodeDetector()
url_data, bbox, straight_qrcode = detect.detectAndDecode(img)
if url_data:
    file = open("business_card_url.txt", "w+")
    file.write(url_data)
    file.close()
    webbrowser.open(url_data)
