# prompt: In the above code instead of gradio use streamlit

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import streamlit as st


# default: Load the model on the available device(s)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto"
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")


def inference(image, search_query=""):
  messages = [
      {
          "role": "user",
          "content": [
              {
                  "type": "image",
                  "image": image,
              },
              {"type": "text", "text": "Extract text from this image."},
          ],
      }
  ]

  # Preparation for inference
  text = processor.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )
  image_inputs, video_inputs = process_vision_info(messages)
  inputs = processor(
      text=[text],
      images=image_inputs,
      padding=True,
      return_tensors="pt",
  )
  inputs = inputs.to("cuda").to(device)
  # Inference: Generation of the output
  generated_ids = model.generate(**inputs, max_new_tokens=200)
  generated_ids_trimmed = [
      out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
  ]
  output_text = processor.batch_decode(
      generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,format="text"
  )

  extracted_text = output_text[0]

  if search_query:
    highlighted_text = extracted_text.replace(
        search_query, f"<span style='background-color:red'>{search_query}</span>"
    )
    return highlighted_text
  else:
    return extracted_text


def main():
    st.title("Image Text Extraction with Search")
    st.write("Upload an image to extract text from it using Qwen2-VL-2B-Instruct. You can then search within the extracted text.")

    uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        search_query = st.text_input("Enter search word...")

        extracted_text = inference(uploaded_image, search_query)

        st.markdown(extracted_text, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
