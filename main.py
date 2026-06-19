# This is a sample Python script.
from transformers import Qwen2VLForConditionalGeneration
from transformers import AutoProcessor
from datasets import load_dataset, get_dataset_config_names
from huggingface_hub import hf_hub_download
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    MODEL_PATH = r"C:\Models\Qwen2-VL-2B-Instruct"

    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_PATH
    )

    print("Model loaded")

    print(get_dataset_config_names("abdoelsayed/CORU"))
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
