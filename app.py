import os
from flask import Flask, request, render_template, send_file
from PIL import Image
import shutil
import uuid
import albumentations as A
import cv2
import numpy as np
import zipfile

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'imageFolder' not in request.files:
        return 'No image folder part in the request'
    if 'annotationFolder' not in request.files:
        return 'No annotation folder part in the request'
    if 'augmentationType' not in request.form:
        return 'No augmentationType in the request'

    # フォルダを受け付ける
    image_folder = request.files.getlist('imageFolder')
    if not image_folder:
        return 'No folder selected'
    annotation_folder = request.files.getlist('annotationFolder')
    if not annotation_folder:
        return 'No folder selected'

    # アノテーション形式を受け付ける
    annotation_type = request.form.get('annotationType')
    if not annotation_type:
        return 'No annotation type selected'

    # データ拡張の種類を受け付ける
    augmentation_type = request.form.get('augmentationType')
    if not augmentation_type:
        return 'No augmentation type selected'

    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    user_id = str(uuid.uuid4())  # アップロードごとに一意のIDを生成
    output_folder = output_folder+'/'+user_id

    os.makedirs(output_folder, exist_ok=True)

    # ここで画像の加工を行います（必要に応じて）-------------------------------------------
    if augmentation_type == "flip":
        transform = A.Compose([A.Flip(p=1)], bbox_params=A.BboxParams(format=annotation_type, min_area=1024,
                                                                      min_visibility=0.1, label_fields=['class_labels']))
    elif augmentation_type == "crop":
        cropHeight = int(request.form.get('cropHeight'))
        cropWidth = int(request.form.get('cropWidth'))
        transform = A.Compose([A.RandomCrop(height=cropHeight, width=cropWidth, p=1)], bbox_params=A.BboxParams(format=annotation_type, min_area=1024,
                                                                                                                min_visibility=0.1, label_fields=['class_labels']))
    elif augmentation_type == "rotate":
        transform = A.Compose([A.Rotate(p=1)], bbox_params=A.BboxParams(format=annotation_type, min_area=1024,
                                                                        min_visibility=0.1, label_fields=['class_labels']))
    elif augmentation_type == "brightness":
        brightnessMin = float(request.form.get('brightnessMin'))
        brightnessMax = float(request.form.get('brightnessMax'))
        transform = A.Compose([A.RandomBrightness(limit=(brightnessMin, brightnessMax), p=1)], bbox_params=A.BboxParams(format=annotation_type, min_area=1024,
                                                                                                                        min_visibility=0.1, label_fields=['class_labels']))
    else:
        transform = A.Compose([A.ShiftScaleRotate(p=1)], bbox_params=A.BboxParams(format=annotation_type, min_area=1024,
                                                                                  min_visibility=0.1, label_fields=['class_labels']))

    # --------------------------------------------------------------------------
    image_folder = sorted(image_folder, key=lambda x: x.filename)
    annotation_folder = sorted(annotation_folder, key=lambda x: x.filename)

    # print(image_folder)

    for file, annotation_file in zip(image_folder, annotation_folder):
        if file.filename == '':
            continue

        img_array = np.asarray(bytearray(file.stream.read()), dtype=np.uint8)
        image = cv2.imdecode(img_array, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # アノテーションファイルを一時ファイルに保存
        annotation_file.save('temp_annotation.txt')
        annotation_file_path = 'temp_annotation.txt'

        # アノテーションファイルを読み込み
        with open(annotation_file_path, 'r') as f:
            anno_lists = f.readlines()

        # 空の配列を用意
        bboxes = []
        class_labels = []

        # アノテーションファイルをパイプラインに与える形に変換
        for anno_list in anno_lists:
            # スペースで区切る
            anno_list = anno_list.split()

            # 0番目の要素からクラスラベルを取り出し
            class_labels.append(anno_list[0])

            # BBoxの値をfloatに変換
            anno_list = [float(i) for i in anno_list[1:]]

            # BBoxの配列に追加
            bboxes.append(anno_list)

        processed = transform(image=image, bboxes=bboxes,
                              class_labels=class_labels)

        filename = file.filename.rsplit('.', 1)[0]  # 拡張子を除いたファイル名を取得
        folder_name = os.path.dirname(file.filename)
        output_folder_path = os.path.join(output_folder, folder_name)
        os.makedirs(output_folder_path, exist_ok=True)

        os.makedirs(output_folder_path+"/image", exist_ok=True)
        os.makedirs(output_folder_path+"/label", exist_ok=True)

        output_image_path = os.path.join(
            output_folder_path+"/image", f'{os.path.basename(filename)}_processed.png')
        output_label_path = os.path.join(
            output_folder_path+"/label", f'{os.path.basename(filename)}_processed.txt')

        processed_image = processed['image']
        processed_bboxes = processed['bboxes']
        processed_class_labels = processed['class_labels']

        # RGBからBGRへ変更
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_image_path, processed_image)

        # Annotationファイルを書き出し
        f = open(output_label_path, 'w')

        for i, processed_bbox in enumerate(processed_bboxes):
            f.write("{} {} {} {} {}\n".format(processed_class_labels[i], processed_bbox[0],
                                              processed_bbox[1], processed_bbox[2], processed_bbox[3]))

        f.close()
        # 一時ファイルを削除
        os.remove(annotation_file_path)

    # 変換後の画像をzipファイルに圧縮してダウンロード
    zip_filename = augmentation_type+'_images.zip'
    zip_path = os.path.join(output_folder, zip_filename)
    shutil.make_archive(os.path.splitext(zip_path)[0], 'zip', output_folder)

    return send_file(f"{os.path.splitext(zip_path)[0]}.zip", as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
