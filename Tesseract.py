import layoutparser as lp
import pytesseract
from PIL import ImageFont
import cv2

# Load the image
image_path = "/home/llmadmin/farz_llm/parser/layout-parser/examples/data/example-table.jpeg"
image = cv2.imread(image_path)
image = image[..., ::-1]

# Load the model
model = lp.Detectron2LayoutModel(
    'lp://PubLayNet/mask_rcnn_X_101_32x8d_FPN_3x/config',
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.65],
    label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"}
)

# Detect layout
layout = model.detect(image)
lp.draw_box(image, layout)

# Extract text blocks
text_blocks = lp.Layout([b for b in layout if b.type == "Text"])
figure_blocks = lp.Layout([b for b in layout if b.type == "Figure"])
text_blocks = lp.Layout([b for b in text_blocks if not any(b.is_in(b_fig) for b_fig in figure_blocks)])

# Sort text blocks by their top-left corner coordinates (reading order)
text_blocks.sort(key=lambda b: (b.coordinates[1], b.coordinates[0]))

# OCR using Tesseract
ocr_agent = lp.TesseractAgent(languages='eng')
for block in text_blocks:
    segment_image = block.pad(left=5, right=5, top=5, bottom=5).crop_image(image)
    text = ocr_agent.detect(segment_image)
    block.set(text=text, inplace=True)

# Debug: Print each block's text
print("Debug: Text blocks content")
for i, block in enumerate(text_blocks):
    print(f"Block {i}: {block.text}")

# Output the text to console
print("Extracted text:")
for txt in text_blocks.get_texts():
    print(txt)
    print('---')

# Save the text to a file
with open("/home/llmadmin/farz_llm/parser/layout-parser/extracted_text.txt", "w") as file:
    for txt in text_blocks.get_texts():
        file.write(txt + '\n---\n')
print("Text has been saved to /home/llmadmin/farz_llm/parser/layout-parser/extracted_text.txt")
