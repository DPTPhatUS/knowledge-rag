from PIL import Image
from transformers import NougatProcessor, VisionEncoderDecoderModel
from huggingface_hub import hf_hub_download
import torch
import time
from pdf2image import convert_from_path

processor = NougatProcessor.from_pretrained("facebook/nougat-small")
model = VisionEncoderDecoderModel.from_pretrained("facebook/nougat-small")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# image = Image.open("../test_data/math/math-004.png").convert("RGB")
# image = Image.open("../test_data/ilovepdf_pages-to-jpg/test_pages-to-jpg-0015.jpg").convert("RGB")
# image = Image.open("../test_data/test.png").convert("RGB")
image = convert_from_path(
    "../test_data/paper.pdf",
    fmt="jpeg",
    first_page=2,
    last_page=2,
)[0].convert("RGB")

# filepath = hf_hub_download(repo_id="hf-internal-testing/fixtures_docvqa", filename="nougat_paper.png", repo_type="dataset")
# image = Image.open(filepath)

start = time.time()

pixel_values = processor(image, return_tensors="pt").pixel_values
# pixel_values = pixel_values.to(dtype=torch.float32)

outputs = model.generate(
    pixel_values.to(device),
    max_new_tokens=2048,
    bad_words_ids=[[processor.tokenizer.unk_token_id]],
)

end = time.time()

sequence = processor.batch_decode(outputs, skip_special_tokens=True)[0]
sequence = processor.post_process_generation(sequence, fix_markdown=False)
print("\n" + sequence)

elapsed = end - start
print("Time elapsed: ", elapsed)
