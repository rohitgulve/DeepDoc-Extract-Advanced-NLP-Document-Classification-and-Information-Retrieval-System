from PIL import Image
from detectron2.utils.logger import setup_logger
from util.model import Model
from flask import Flask, jsonify, request
from Logger.Logger import logger
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.visualizer import Visualizer
import requests
import json
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
                # file = request.files
                filename = file.filename
                filename = filename.lower()
                logger.info("filename: {}".format(filename))
                result = {}
                # res1 = request.data['result']
                # print(res1)
                print('filename', filename)
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
                                data = data.encode('ascii', 'ignore').decode("utf-8")
                                stop_words = set(stopwords.words('english'))
                                word_tokens = word_tokenize(data)
                                data = " ".join(w for w in word_tokens if not w in stop_words)
                                pred_data[cls] = data
                        if 'total' in pred_data:
                            total = pred_data.get('total')
                            # to_regex = re.search('[$0-9]+,?[0-9]+.[0-9]+', total)
                            pattern = re.compile(
                                "\\b(total|TOTAL|AMOUNT DUE :|Invoice Total :|Invoice Total|AMOUNT :)\\W")
                            to_regex = pattern.sub("", total)
                            # pred_data['total'] = str(to_regex.group())
                            pred_data['total'] = to_regex
                            # print(to_regex.group())
                        if 'invoice_number' in pred_data:
                            invoice_num = pred_data.get('invoice_number')
                            # to_regex = re.search('([A-Z0-9-|\s]+|\s\s)', invoice_num)
                            pattern = re.compile(
                                "\\b(invoiceNo . :|Invoice ID :|nvoice No . :|Invoice Number :|Invoice No "
                                ".|invoiceNo .)\\W")
                            to_regex = pattern.sub("", invoice_num)
                            # to_regex = re.sub('invoiceNo . :', '', invoice_num)
                            pred_data['invoice_number'] = to_regex
                        if 'invoice_date' in pred_data:
                            invoice_date = pred_data.get('invoice_date')
                            # to_regex = re.search('([A-Z0-9-|\s]+|\s\s)', invoice_num)
                            pattern = re.compile("\\b(DATE|Date|Invoice :)\\W")
                            # to_regex = re.sub('Date ', '', invoice_date)
                            to_regex = pattern.sub("", invoice_date)
                            pred_data['invoice_date'] = to_regex
                            #     # print(to_regex.group())
                        if 'shipping_address' in pred_data:
                            shipping_addr = pred_data.get('shipping_address')
                            pattern = re.compile("\\b(Ship To|SHIP TO)\\W")
                            to_regex = pattern.sub("", shipping_addr)
                            # to_regex = re.sub('Ship To ', '', shipping_addr)
                            pred_data['shipping_address'] = to_regex
                            # print(to_regex.group())
                        os.remove(os.path.join(image_path, fname))
                        result[str(i)] = pred_data
                    return jsonify({"result": result, "status": "success"}), 200
                # else:-
                #     return jsonify({"Message": "Inappropriate service passed", "status": "fail"}), 500

                elif filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
                    imagePath = os.path.join(tmp_path, filename)
                    predictor = DefaultPredictor(model)
                    image = cv2.imread(imagePath)
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

                    img = Image.open(imagePath)
                    for j in range(len(pred_boxes)):
                        if scores[j] > 0.9:
                            cls = classes[pred_classes[j]]
                            img2 = img.crop((pred_boxes[j]))
                            # img2.save("./images/cropedImages/" + str(j + 1) + ".jpg")
                            # test = cv2.imread("./images/cropedImages/" + str(j + 1) + ".jpg")
                            # img2.imwrite("./images/cropped.jpg", image)
                            data = pytesseract.image_to_string(img2, lang='eng', config='--psm 6')
                            data = data.encode('ascii', 'ignore').decode("utf-8")
                            print(data)
                            # file = [('file', open("./images/cropedImages/" + str(j + 1) + ".jpg", 'rb'))]
                            # r = requests.request("POST", 'http://172.26.21.10:5004/extractText', files=file)
                            # reponse = r.json()
                            
                            # print(reponse)
                            # data = re.sub(r'[?|.|!]', r'', data)
                            # data = re.sub(r'[^a-zA-Z0-9 ]', r'', data)
                            stop_words = set(stopwords.words('english'))
                            word_tokens = word_tokenize(data)
                            data = " ".join(w for w in word_tokens if not w in stop_words)
                            # words = set(nltk.corpus.words.words()) data = " ".join(w for w in
                            # nltk.wordpunct_tokenize(data) if w.lower() in words or not w.isalpha())
                            pred_data[cls] = data
                    if 'total' in pred_data:
                        total = pred_data.get('total')
                        # to_regex = re.search('[$0-9]+,?[0-9]+.[0-9]+', total)
                        pattern = re.compile("\\b(total|TOTAL|AMOUNT DUE :|Invoice Total :|Invoice Total|AMOUNT :)\\W")
                        to_regex = pattern.sub("", total)
                        # pred_data['total'] = str(to_regex.group())
                        pred_data['total'] = to_regex
                        # print(to_regex.group())
                    if 'invoice_number' in pred_data:
                        invoice_num = pred_data.get('invoice_number')
                        # to_regex = re.search('([A-Z0-9-|\s]+|\s\s)', invoice_num)
                        pattern = re.compile(
                            "\\b(invoiceNo . :|Invoice ID :|nvoice No . :|Invoice Number :|Invoice No .|invoiceNo .)\\W")
                        to_regex = pattern.sub("", invoice_num)
                        # to_regex = re.sub('invoiceNo . :', '', invoice_num)
                        pred_data['invoice_number'] = to_regex
                    if 'invoice_date' in pred_data:
                        invoice_date = pred_data.get('invoice_date')
                        # to_regex = re.search('([A-Z0-9-|\s]+|\s\s)', invoice_num)
                        pattern = re.compile("\\b(DATE|Date|Invoice :)\\W")
                        # to_regex = re.sub('Date ', '', invoice_date)
                        to_regex = pattern.sub("", invoice_date)
                        pred_data['invoice_date'] = to_regex
                    #     # print(to_regex.group())
                    if 'shipping_address' in pred_data:
                        shipping_addr = pred_data.get('shipping_address')
                        pattern = re.compile("\\b(Ship To|SHIP TO)\\W")
                        to_regex = pattern.sub("", shipping_addr)
                        # to_regex = re.sub('Ship To ', '', shipping_addr)
                        pred_data['shipping_address'] = to_regex
                        # print(to_regex.group())

                    return jsonify({"result": pred_data, "status": "success"}), 200
                else:
                    return jsonify({"Message": "Inappropriate service passed", "status": "fail"}), 500

            except Exception as ex:
                logger.exception("Error in input file" + str(ex))
                return jsonify({"Message": "Inappropriate File passed", "status": "fail"}), 500

        elif len(request.values) != 0:
            encodedText= request.values["input"]

            result = {}
            # encodedText = request.data

            print(type(encodedText))
            encodedList=encodedText.strip("][").split(',')



            # print((encodedText))
            # print("*" * 100)
            # print(type(encodedText))
            # output_dict = json.loads(encodedText)  # convert string dictionary to dict format
            for i in range(len(encodedList)):
                res = encodedList[i]
                res = (res).replace("b'", "")
                res = (res).replace("'", "")
                # print(((res)))
                # print("$" * 100)
                res = res.encode("utf_8")

            # image = '/home/rgulve/Documents/ProjectWork/InvoiceDataExtraction/input_files/1.jpg'
            # import base64
            # with open(image, "rb") as img_file:
            #     my_string = base64.b64encode(img_file.read())
            # print(my_string)

                import base64
                imgdata = base64.b64decode(res)
                filename = './usecaseData/result'+str(i)+'.jpg'  # I assume you have a way of picking unique filenames
                with open(filename, 'wb') as f:
                    f.write(imgdata)

                # imagePath = os.path.join(tmp_path, filename)
                predictor = DefaultPredictor(model)
                image = cv2.imread(filename)
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

                img = Image.open(filename)
                for j in range(len(pred_boxes)):
                    if scores[j] > 0.9:
                        cls = classes[pred_classes[j]]
                        img2 = img.crop((pred_boxes[j]))
                        # img2.save("./images/cropedImages/" + str(j + 1) + ".jpg")
                        # test = cv2.imread("./images/cropedImages/" + str(j + 1) + ".jpg")
                        # img2.imwrite("./images/cropped.jpg", image)
                        data = pytesseract.image_to_string(img2, lang='eng', config='--psm 6')
                        data = data.encode('ascii', 'ignore').decode("utf-8")
                        print(data)
                        # file = [('file', open("./images/cropedImages/" + str(j + 1) + ".jpg", 'rb'))]
                        # r = requests.request("POST", 'http://172.26.21.10:5004/extractText', files=file)
                        # reponse = r.json()
                        # print(reponse)
                        # data = re.sub(r'[?|.|!]', r'', data)
                        # data = re.sub(r'[^a-zA-Z0-9 ]', r'', data)
                        stop_words = set(stopwords.words('english'))
                        word_tokens = word_tokenize(data)
                        data = " ".join(w for w in word_tokens if not w in stop_words)
                        # words = set(nltk.corpus.words.words()) data = " ".join(w for w in
                        # nltk.wordpunct_tokenize(data) if w.lower() in words or not w.isalpha())
                        pred_data[cls] = data
                if 'total' in pred_data:
                    total = pred_data.get('total')
                    # to_regex = re.search('[$0-9]+,?[0-9]+.[0-9]+', total)
                    pattern = re.compile("\\b(total|TOTAL|AMOUNT DUE :|Invoice Total :|Invoice Total|AMOUNT :)\\W")
                    to_regex = pattern.sub("", total)
                    # pred_data['total'] = str(to_regex.group())
                    pred_data['total'] = to_regex
                    # print(to_regex.group())
                if 'invoice_number' in pred_data:
                    invoice_num = pred_data.get('invoice_number')
                    # to_regex = re.search('([A-Z0-9-|\s]+|\s\s)', invoice_num)
                    pattern = re.compile(
                        "\\b(invoiceNo . :|Invoice ID :|nvoice No . :|Invoice Number :|Invoice No .|invoiceNo .)\\W")
                    to_regex = pattern.sub("", invoice_num)
                    # to_regex = re.sub('invoiceNo . :', '', invoice_num)
                    pred_data['invoice_number'] = to_regex
                if 'invoice_date' in pred_data:
                    invoice_date = pred_data.get('invoice_date')
                    # to_regex = re.search('([A-Z0-9-|\s]+|\s\s)', invoice_num)
                    pattern = re.compile("\\b(DATE|Date|Invoice :)\\W")
                    # to_regex = re.sub('Date ', '', invoice_date)
                    to_regex = pattern.sub("", invoice_date)
                    pred_data['invoice_date'] = to_regex
                #     # print(to_regex.group())
                if 'shipping_address' in pred_data:
                    shipping_addr = pred_data.get('shipping_address')
                    pattern = re.compile("\\b(Ship To|SHIP TO)\\W")
                    to_regex = pattern.sub("", shipping_addr)
                    # to_regex = re.sub('Ship To ', '', shipping_addr)
                    pred_data['shipping_address'] = to_regex
                    # print(to_regex.group())
                result[str(i)] = pred_data
                print('lenght of result',len(result))
            return jsonify({"result": result, "status": "success"}), 200

        else:
            return jsonify({"Message": "Images with extension {'jpg', 'jpeg', 'pdf', 'png'} are allowed", "status": "fail"}), 500


if __name__ == '__main__':
    import nltk
    from pathlib import Path

    Path("usecaseData").mkdir(parents=True, exist_ok=True)
    Path("images").mkdir(parents=True, exist_ok=True)

    nltk.download('stopwords')
    nltk.download('punkt')
    model = Model.model()
    # model = globalVariable.model
    # model.MODEL.WEIGHTS = '/InvoiceDataExtraction/output/model_final.pth'  # directory and file name
    model.MODEL.WEIGHTS = os.path.join('output', "model_final.pth")  # directory and file name
    model.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    app.run(debug=True, host='0.0.0.0')
