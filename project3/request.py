import requests
import base64
import os

def png_to_dataurl(filepath: str) -> str:
    # Source - https://stackoverflow.com/a
    # Posted by Jon, modified by community. See post 'Timeline' for change history
    # Retrieved 2026-01-14, License - CC BY-SA 4.0

    binary_fc = open(filepath, 'rb').read()  # fc aka file_content
    base64_utf8_str = base64.b64encode(binary_fc).decode('utf-8')

    ext = filepath.split('.')[-1]
    return f'data:image/{ext};base64,{base64_utf8_str}'


if __name__ == "__main__":
    #print(png_to_dataurl("./example.png"))
    ip = os.environ.get("AWS_IP")
    if ip == None:
        print ("No AWS_IP env variable set")
        exit(1)
    resp = requests.post(
        f"http://{ip}:8000/predict",
        json={"input": png_to_dataurl("example.png")}  # adapt to your model format
    )
    print(resp.json())