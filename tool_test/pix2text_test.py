from pix2text import Pix2Text

img_fp = "../test_data/math.pdf"
p2t = Pix2Text.from_config()
doc = p2t.recognize_pdf(img_fp, page_numbers=[41])
doc.to_markdown(
    "output-md"
)  # The exported Markdown information is saved in the output-md directory
