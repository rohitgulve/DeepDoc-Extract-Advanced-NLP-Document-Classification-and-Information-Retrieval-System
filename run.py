from PIL import Image
from detectron2.utils.logger import setup_logger
from util.model import Model
from flask import Flask, jsonify, request
from Logger.Logger import logger
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer

setup_logger()
import warnings
import pytesseract
from pdf2image import convert_from_path

warnings.filterwarnings('ignore')
import cv2
from detectron2.engine import DefaultPredictor
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

tmp_path = "./input_files/"
image_path = "./pdf2images/"

ALLOWED_TRAINUPLOAD_EXTENSIONS = {'zip'}
ALLOWED_PREDUPLOAD_EXTENSIONS = {'jpg', 'jpeg', 'png', 'pdf'}
app = Flask(__name__)


def allowed_file(filename, param='PREDICTION'):
    if param == "UPLOAD":
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_TRAINUPLOAD_EXTENSIONS
    else:
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in ALLOWED_PREDUPLOAD_EXTENSIONS


@app.route("/train", methods=['GET', 'POST'])
def train():
    if request.method == 'POST':

        try:
            #     file = request.files['file']
            #     # filename = request.files['file'].filename
            #     if file and allowed_file(file.filename, 'UPLOAD'):
            #         with zipfile.ZipFile(file, 'r') as zip_ref:
            #             zip_ref.extractall(UPLOAD_FOLDER)

            Model.trainModel()
            # #return jsonify(
            #     {"Message": "Please upload the appropriate file with extension '.zip",
            #      "status": "Fail"}), 500

            # Model.trainModel()
            return jsonify({"Message": "Model has been trained successfully", "status": "success"}), 200

        except Exception as ex:
            logger.exception("Error while training...." + str(ex))
            return jsonify({"Message": "Error while training....", "status": "fail"}), 500

    else:
        return jsonify({"Message": "Inappropriate service passed", "status": "fail"}), 500


@app.route("/prediction", methods=['GET', 'POST'])
def run():
    if request.method == 'POST':

        if (len(request.files) != 0) and allowed_file(request.files['file'].filename):
            try:
                file = request.files['file']
                filename = file.filename
                filename = filename.lower()
                logger.info("filename: {}".format(filename))
                file.save(os.path.join(tmp_path, filename))
                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(tmp_path, filename)
                    images = convert_from_path(pdf_path)
                    res_out = []
                    for i, image in enumerate(images):
                        fname = "image" + str(i) + ".png"
                        image.save(os.path.join(image_path, fname))
                        input_image = os.path.join(image_path, fname)
                        predictor = DefaultPredictor(model)
                        image = cv2.imread(input_image)
                        outputs = predictor(image)
                        v = Visualizer(image, scale=0.8, instance_mode=ColorMode.IMAGE)
                        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                        pred_boxes = outputs.get("instances").pred_boxes.tensor.tolist()  # Bounding boxes
                        pred_classes = outputs.get("instances").pred_classes.tolist()  # Bounding boxes
                        scores = outputs.get("instances").scores.tolist()  # Bounding boxes
                        print(scores)

                        logger.info("Number of fields found:- {}".format(len(pred_boxes)))  # print count bounding box
                        pred_data = {}
                        classes = ["invoice_date", "invoice_number", "shipping_address", "total", "vendor_address",
                                   "vendor_name"]

                        img = Image.open(input_image)
                        for j in range(len(pred_boxes)):
                            if scores[j] > 0.9:
                                cls = classes[pred_classes[j]]
                                img2 = img.crop((pred_boxes[j]))
                                data = pytesseract.image_to_string(img2, lang='eng', config='--psm 6')
                                data = data.replace('\n', '')
                                data = data.replace('\f', '')

                                pred_data[cls] = data
                                print(data)
                        os.remove(os.path.join(image_path, fname))
                        pred_data['page_no'] = i + 1
                        res_out.append(pred_data)
                    return jsonify({"result": pred_data, "status": "success"}), 200
                # else:-
                #     return jsonify({"Message": "Inappropriate service passed", "status": "fail"}), 500

                elif filename.endswith('.jpg') or filename.endswith('.png'):
                    imagePath = os.path.join(tmp_path, filename)
                    predictor = DefaultPredictor(model)
                    image = cv2.imread(imagePath)
                    outputs = predictor(image)
                    v = Visualizer(image, scale=0.8, instance_mode=ColorMode.IMAGE)
                    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
                    pred_boxes = (outputs.get("instances").pred_boxes).tensor.tolist()  # Bounding boxes
                    pred_classes = (outputs.get("instances").pred_classes).tolist()  # Bounding boxes
                    scores = (outputs.get("instances").scores).tolist()  # Bounding boxes
                    print(scores)

                    logger.info("Number of fields found:- {}".format(len(pred_boxes)))  # print count bounding box
                    pred_data = {}
                    classes = ["invoice_date", "invoice_number", "shipping_address", "total", "vendor_address",
                               "vendor_name"]

                    img = Image.open(imagePath)
                    for j in range(len(pred_boxes)):
                        if scores[j] > 0.9:
                            cls = classes[pred_classes[j]]
                            img2 = img.crop((pred_boxes[j]))
                            data = pytesseract.image_to_string(img2, lang='eng', config='--psm 6')
                            # data = re.sub(r'[?|.|!]', r'', data)
                            # data = re.sub(r'[^a-zA-Z0-9 ]', r'', data)
                            stop_words = set(stopwords.words('english'))
                            word_tokens = word_tokenize(data)
                            data = " ".join(w for w in word_tokens if not w in stop_words)
                            # words = set(nltk.corpus.words.words())
                            # data = " ".join(w for w in nltk.wordpunct_tokenize(data) if w.lower() in words or not w.isalpha())
                            pred_data[cls] = data
                    if 'total' in pred_data:
                        total = pred_data.get('total')
                        to_regex = re.search('[$0-9]+,?[0-9]+.[0-9]+', total)
                        pred_data['total'] = str(to_regex.group())
                        # print(to_regex.group())
                    # if 'invoice_number' in pred_data:
                    #     invoice_num = pred_data.get('invoice_number')
                    #     to_regex = re.search('([A-Z0-9-|\s]+|\s\s)', invoice_num)
                    #     pred_data['invoice_number'] = str(to_regex.group())
                    #     # print(to_regex.group())
                    if 'shipping_address' in pred_data:
                        shipping_addr = pred_data.get('shipping_address')
                        to_regex = re.sub('Ship To', '', shipping_addr)
                        pred_data['shipping_address'] = to_regex
                    #     # print(to_regex.group())

                    return jsonify({"result": pred_data, "status": "success"}), 200
                else:
                    return jsonify({"Message": "Inappropriate service passed", "status": "fail"}), 500

            except Exception as ex:
                logger.exception("Error in input file" + str(ex))
                return jsonify({"Message": "Inappropriate File passed", "status": "fail"}), 500
        else:
            return jsonify({"Message": "Images with extention {'jpg', 'jpeg', 'pdf','png'} are allowed", "status": "fail"}), 500


if __name__ == '__main__':
    model = Model.model()
    # model = globalVariable.model
    model.MODEL.WEIGHTS = './output/model_final.pth'  # directory and file name
    # model.MODEL.WEIGHTS = os.path.join('output', "model_final.pth")  # directory and file name
    model.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    app.run(debug=False, host='172.30.24.27', port=5006)
