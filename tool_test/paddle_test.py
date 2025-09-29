import numpy as np
import paddle
from paddleocr import PaddleOCR, FormulaRecognition
import time
from pdf2image import convert_from_path

gpu_available = paddle.device.is_compiled_with_cuda()
print("GPU available:", gpu_available)

# ocr = PaddleOCR(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False,
#     use_textline_orientation=False,
# )

# start = time.time()

# images = convert_from_path(
#     "../test_data/math.pdf",
#     fmt="png",
#     first_page=41,
#     last_page=41,
# )
# result = ocr.predict(input=[np.array(image.convert("RGB")) for image in images[:1]])

# # result = ocr.predict(
# #     input=[
# #         f"../test_data/ilovepdf_pages-to-jpg/test_pages-to-jpg-{i:04}.jpg"
# #         for i in range(1, 119)
# #     ]
# # )

# end = time.time()

# for res in result:
#     res.print()
#     res.save_to_img("output")
#     res.save_to_json("output")

# elapsed = end - start
# print("Time elapsed: ", elapsed)


model = FormulaRecognition()

start = time.time()

output = model.predict(
    input=[
        "../test_data/math/math-041.png"
    ]
)

end = time.time()

for res in output:
    res.print()
    res.save_to_img(save_path="output")
    res.save_to_json(save_path="output")

elapsed = end - start
print("Time elapsed: ", elapsed)
