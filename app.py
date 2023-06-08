import os
from flask import Flask, request, render_template, send_file
from PIL import Image
import shutil

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

    for file in folder:
        if file.filename == '':
            continue

        image = Image.open(file)
        # ここで画像の加工を行います（必要に応じて）
        processed_image = image.convert('L')  # 例：画像をグレースケールに変換

        filename = file.filename.rsplit('.', 1)[0]  # 拡張子を除いたファイル名を取得
        folder_name = os.path.dirname(file.filename)
        output_folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        output_path = os.path.join(
            output_folder_path, f'{os.path.basename(filename)}_processed.png')
        processed_image.save(output_path, 'PNG')

    # 変換後の画像をzipファイルに圧縮してダウンロード
    zip_filename = 'processed_images.zip'
    zip_path = os.path.join(output_folder, zip_filename)
    shutil.make_archive(os.path.splitext(zip_path)[0], 'zip', output_folder)

    return send_file(f"{os.path.splitext(zip_path)[0]}.zip", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
