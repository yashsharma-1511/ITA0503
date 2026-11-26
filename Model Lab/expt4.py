import cv2
import pytesseract

# Set Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

cap = cv2.VideoCapture("output_reversed.mp4")

file = open("output.txt", "w")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Extract text using OCR
    text = pytesseract.image_to_string(gray)

    if text.strip():
        print(text)
        file.write(text + "\n")

        # Display recognized text
        cv2.putText(frame, text.strip().split('\n')[0],
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 0), 2)

    cv2.imshow("Video OCR", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

file.close()
cap.release()
cv2.destroyAllWindows()