import os
from flask import Flask, request, render_template, send_file
from PIL import Image
import shutil
import uuid
import albumentations as A
import cv2
import numpy as np

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'folder' not in request.files:
        return 'No folder part in the request'

    folder = request.files.getlist('folder')
    if not folder:
        return 'No folder selected'

    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    user_id = str(uuid.uuid4())  # アップロードごとに一意のIDを生成
    output_folder = output_folder+'/'+user_id

    os.makedirs(output_folder, exist_ok=True)

    for file in folder:
        if file.filename == '':
            continue

        img_array = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, 1)
        # image = cv2.imread(str(file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # ここで画像の加工を行います（必要に応じて）
        transform = A.Compose([A.Flip(p=1)])
        # processed_image = image.convert('L')  # 例：画像をグレースケールに変換
        processed = transform(image=image)

        filename = file.filename.rsplit('.', 1)[0]  # 拡張子を除いたファイル名を取得
        folder_name = os.path.dirname(file.filename)
        output_folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        output_path = os.path.join(
            output_folder_path, f'{os.path.basename(filename)}_processed.png')
        processed_image = processed['image']

        # RGBからBGRへ変更
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, processed_image)

    # 変換後の画像をzipファイルに圧縮してダウンロード
    zip_filename = 'processed_images.zip'
    zip_path = os.path.join(output_folder, zip_filename)
    shutil.make_archive(os.path.splitext(zip_path)[0], 'zip', output_folder)

    return send_file(f"{os.path.splitext(zip_path)[0]}.zip", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
