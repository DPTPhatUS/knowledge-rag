import os
from pdf2image import convert_from_path
import time

start = time.time()
images = convert_from_path("../test_data/test.pdf")
end = time.time()
elapsed = end - start
print("Time elapsed: ", elapsed)

save_dir = "images"
os.makedirs(save_dir, exist_ok=True)

for index, image in enumerate(images):
    image = image.convert("RGB")
    image.save(f"{save_dir}/test_{index}.png")
