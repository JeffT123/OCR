from paddleocr import PaddleOCR, draw_ocr
import fitz
from PIL import Image, ImageFont
import cv2
import numpy as np
import os
import layoutparser as lp

# Path to the PDF file
pdf_path = '/home/llmadmin/farz_llm/Contents_2006_Advances-in-Virus-Research.pdf'

# Check the number of pages in the PDF
with fitz.open(pdf_path) as pdf:
    num_pages = pdf.page_count

# Set PAGE_NUM to the total number of pages in the PDF
PAGE_NUM = min(10, num_pages)
print(f'Processing {PAGE_NUM} pages of the PDF.')

# Initialize OCR with English language support
ocr = PaddleOCR(use_angle_cls=True, lang="en", page_num=PAGE_NUM)

# Perform OCR on the PDF
result = ocr.ocr(pdf_path, cls=True)
for idx in range(len(result)):
    res = result[idx]
    if res is None:
        print(f"[DEBUG] Empty page {idx+1} detected, skip it.")
        continue
    for line in res:
        print(line)

# Extract images from PDF and prepare for visualization
imgs = []
with fitz.open(pdf_path) as pdf:
    for pg in range(0, PAGE_NUM):
        page = pdf[pg]
        mat = fitz.Matrix(2, 2)
        pm = page.get_pixmap(matrix=mat, alpha=False)
        if pm.width > 2000 or pm.height > 2000:
            pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)
        img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        imgs.append(img)

# Initialize LayoutParser model
model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    model_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/model",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# Create the output directory if it doesn't exist
output_dir = '/home/llmadmin/farz_llm/output6'
os.makedirs(output_dir, exist_ok=True)

# Path to a common system font
font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'

# Save the annotated images in the specified output directory
for idx in range(len(result)):
    res = result[idx]
    if res is None:
        continue
    image = imgs[idx]
    
    # Detect layout elements
    layout = model.detect(image)
    table_blocks = lp.Layout([b for b in layout if b.type == 'Table'])
    
    # Annotate tables
    image_with_tables = lp.draw_box(image, table_blocks, box_width=3, box_color='red')

    # Perform OCR on detected table regions
    for block in table_blocks:
        x1, y1, x2, y2 = map(int, block.coordinates)
        table_image = image[y1:y2, x1:x2]
        
        # Perform OCR on the table region
        table_results = ocr.ocr(table_image, cls=True)
        if table_results:
            for line in table_results[0]:
                box = [tuple(point) for point in line[0]]
                box = [(min(point[0] for point in box) + x1, min(point[1] for point in box) + y1),
                       (max(point[0] for point in box) + x1, max(point[1] for point in box) + y1)]
                txt = line[1][0]
                cv2.rectangle(image_with_tables, box[0], box[1], color=(0, 255, 0), thickness=2)
                cv2.putText(image_with_tables, txt, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Save annotated image
    output_image_path = os.path.join(output_dir, f'result_page_{idx+1}.jpg')
    cv2.imwrite(output_image_path, image_with_tables)

# Create the output directory if it doesn't exist
output_dir = '/home/llmadmin/farz_llm/output6'
os.makedirs(output_dir, exist_ok=True)

# Define the path for the extracted text file
text_output_file = os.path.join(output_dir, 'extracted_text.txt')

extracted_texts = []

# OCR processing loop
for idx in range(len(result)):
    res = result[idx]
    if res is None:
        print(f"[DEBUG] Empty page {idx+1} detected, skip it.")
        continue
    page_text = ""
    for line in res:
        print(line)
        page_text += line[1][0] + "\n"
    extracted_texts.append(page_text)

# Write the extracted texts to the file
with open(text_output_file, 'w') as f:
    for page_text in extracted_texts:
        f.write(page_text + "\n")
