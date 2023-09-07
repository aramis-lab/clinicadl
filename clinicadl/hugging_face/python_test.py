# from clinicadl.hugging_face.hugging_face import load_from_hf_hub

# load_from_hf_hub("test","camillebri/test654646",)
# https://huggingface.co/camillebri/test654646/tree/main


def main_process(callback):
    print("Main process started.")
    print("Doing some tasks.")
    result = "Task completed."
    callback(result)  # Call the callback function with the result
    print("Main process finished.")


def my_callback_function(message):
    print(f"Inside the callback: {message}")


# Run the example
main_process(None)
