# copyright (c) 2022 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import tarfile
import os
import time
import datetime
import functools
import copy
import cv2
import platform
import numpy as np
import fitz
import traceback
from PIL import Image
from pdf2docx.converter import Converter
from qtpy.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar, \
                           QGridLayout, QMessageBox, QLabel, QFileDialog, QCheckBox, \
                           QTextEdit, QScrollArea
from qtpy.QtCore import Signal, QThread, QObject, QTimer
from qtpy.QtGui import QImage, QPixmap, QIcon

file = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(file, '../../'))
sys.path.append(file)
sys.path.insert(0, root)

from ppstructure.predict_system import StructureSystem, save_structure_res
from ppstructure.utility import parse_args, draw_structure_result
from ppocr.utils.network import download_with_progressbar
from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx, convert_info_docx_multi_page
# from ScreenShotWidget import ScreenShotWidget

__APPNAME__ = "pdf2word"
__VERSION__ = "1.0.0"


def redink_remover(img):
    """
    去红章识别：从图像中移除红色区域，保留其他颜色信息
    将红色通道复制到三个通道，用于红章识别前的预处理
    """
    image = copy.deepcopy(img)
    blue_c, green_c, red_c = cv2.split(image)
    result_img = np.expand_dims(red_c, axis=2)
    result_img = np.concatenate((result_img, result_img, result_img), axis=-1)
    return result_img


def get_available_docx_path(docx_file, max_retries=5):
    """
    处理文件锁定问题，并且在文件已存在时自动改改文件名
    或者等待文件被释放
    """
    base_path = docx_file.rsplit('.', 1)[0]  # 不含后缀的路径
    ext = '.docx'
    
    # 先尝试处理文件锁定
    for attempt in range(max_retries):
        if not os.path.exists(docx_file):
            return docx_file
        
        try:
            # 尝试删除文件，如果成功说明文件未被锁定
            os.remove(docx_file)
            print(f"Successfully removed existing file: {docx_file}")
            return docx_file
        except PermissionError:
            if attempt < max_retries - 1:
                print(f"File is locked, waiting... (attempt {attempt + 1}/{max_retries})")
                time.sleep(1)  # 等待一秒后重试
            else:
                # 如果一直无法删除，改为自动改改文件名
                print(f"File remains locked after {max_retries} retries, generating new filename...")
                counter = 1
                while True:
                    new_filename = f"{base_path}_{counter}{ext}"
                    if not os.path.exists(new_filename):
                        print(f"Using alternative filename: {new_filename}")
                        return new_filename
                    counter += 1
        except Exception as e:
            print(f"Error processing file: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    
    # 万一仍然失败，返回原始路径，由上层处理
    return docx_file


URLs_EN = {
    # 下载超英文轻量级PP-OCRv3模型的检测模型并解压
    "en_PP-OCRv3_det_infer":
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
    # 下载英文轻量级PP-OCRv3模型的识别模型并解压
    "en_PP-OCRv3_rec_infer":
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
    # 下载超轻量级英文表格英文模型并解压
    "en_ppstructure_mobile_v2.0_SLANet_infer":
    "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
    # 英文版面分析模型
    "picodet_lcnet_x1_0_fgd_layout_infer":
    "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar",
}
DICT_EN = {
    "rec_char_dict_path": "en_dict.txt",
    "layout_dict_path": "layout_publaynet_dict.txt",
}

URLs_CN = {
    # 下载超中文轻量级PP-OCRv3模型的检测模型并解压
    "cn_PP-OCRv3_det_infer":
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
    # 下载中文轻量级PP-OCRv3模型的识别模型并解压
    "cn_PP-OCRv3_rec_infer":
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
    # 下载超轻量级英文表格英文模型并解压
    "cn_ppstructure_mobile_v2.0_SLANet_infer":
    "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
    # 中文版面分析模型
    "picodet_lcnet_x1_0_fgd_layout_cdla_infer":
    "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar",
}
DICT_CN = {
    "rec_char_dict_path": "ppocr_keys_v1.txt",
    "layout_dict_path": "layout_cdla_dict.txt",
}


def QImageToCvMat(incomingImage) -> np.array:
    '''  
    Converts a QImage into an opencv MAT format  
    '''

    incomingImage = incomingImage.convertToFormat(QImage.Format.Format_RGBA8888)

    width = incomingImage.width()
    height = incomingImage.height()

    ptr = incomingImage.bits()
    ptr.setsize(height * width * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((height, width, 4))
    return arr


def normalize_path(path):
    """
    规范化路径，处理Windows上的中文路径问题
    将路径转换为绝对路径并处理编码
    """
    try:
        # 先将路径转换为绝对路径
        abs_path = os.path.abspath(path)
        # 确保路径存在
        if not os.path.exists(abs_path):
            print(f"Warning: Path does not exist: {abs_path}")
            # 尝试原始路径
            if os.path.exists(path):
                return path
        return abs_path
    except Exception as e:
        print(f"Error normalizing path {path}: {e}")
        return path


def read_image_with_opencv(image_file):
    """
    使用OpenCV读取图像，处理中文路径问题
    Windows上cv2.imread不支持中文路径，所以使用numpy数组方式
    """
    try:
        # 方法1：直接用cv2.imread（对于ASCII路径有效）
        img = cv2.imread(image_file, cv2.IMREAD_COLOR)
        if img is not None:
            return img
        
        # 方法2：如果方法1失败，使用numpy+PIL处理中文路径
        print(f"cv2.imread failed, trying alternative method for: {image_file}")
        from PIL import Image as PILImage
        pil_img = PILImage.open(image_file).convert('RGB')
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        if img is not None:
            print(f"Successfully read image using PIL: {image_file}")
            return img
    except Exception as e:
        print(f"Error reading image with both methods: {image_file}, Error: {e}")
    
    return None

def save_image_with_chinese_path(img, save_path):
    """
    保存图像到中文路径，规避cv2.imwrite的中文路径问题
    使用PIL库来处理中文路径的图像保存
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # 方法1：先用cv2.imwrite尝试保存（如果路径中没有中文会成功）
        result = cv2.imwrite(save_path, img)
        if result:
            print(f"Image saved successfully with cv2: {save_path}")
            return True
        
        # 方法2：如果cv2失败，使用PIL保存（支持中文路径）
        print(f"cv2.imwrite failed, using PIL for: {save_path}")
        from PIL import Image as PILImage
        
        # 将OpenCV的BGR格式转换为PIL的RGB格式
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(img_rgb)
        pil_img.save(save_path, 'JPEG', quality=95)
        print(f"Image saved successfully with PIL: {save_path}")
        return True
    except Exception as e:
        print(f"Error saving image to {save_path}: {type(e).__name__}: {e}")
        return False


def readImage(image_file) -> list:
    """
    读取图像或PDF文件，支持中文路径
    """
    imgs = []  # 初始化为空列表
    try:
        # 规范化路径
        image_file = normalize_path(image_file)
        
        # 检查文件是否存在
        if not os.path.exists(image_file):
            print(f"Error: File does not exist: {image_file}")
            return imgs
        
        # 判断是PDF还是图像
        file_ext = os.path.basename(image_file)[-3:].lower()
        
        if file_ext == 'pdf':
            # 处理PDF文件
            with fitz.open(image_file) as pdf:
                for pg in range(0, pdf.page_count):
                    page = pdf[pg]
                    mat = fitz.Matrix(2, 2)
                    pm = page.get_pixmap(matrix=mat, alpha=False)

                    # if width or height > 2000 pixels, don't enlarge the image
                    if pm.width > 2000 or pm.height > 2000:
                        pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                    img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                    imgs.append(img)
                print(f"Successfully read PDF: {image_file}, pages: {len(imgs)}")
        else:
            # 处理普通图像文件
            img = read_image_with_opencv(image_file)
            if img is not None:
                imgs = [img]
                print(f"[img info] shape: {img.shape}, file: {image_file}")
            else:
                print(f"Warning: Unable to read image file: {image_file}")
    except Exception as e:
        print(f"Error reading image file {image_file}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    
    return imgs



class Worker(QThread):
    progressBarValue = Signal(int)
    progressBarRange = Signal(int)
    endsignal = Signal()
    exceptedsignal = Signal(str)  #发送一个异常信号
    warningsignal = Signal(str)  # 发送警告信号
    loopFlag = True

    def __init__(self, predictors, save_pdf, vis_font_path, use_pdf2docx_api, keep_reference=True):
        super(Worker, self).__init__()
        self.predictors = predictors
        self.save_pdf = save_pdf
        self.vis_font_path = vis_font_path
        self.lang = 'CN'
        self.imagePaths = []
        self.use_pdf2docx_api = use_pdf2docx_api
        self.keep_reference = keep_reference  # 是否保留签批区
        self.outputDir = None
        self.totalPageCnt = 0
        self.pageCnt = 0
        self.setStackSize(1024 * 1024)

    def setImagePath(self, imagePaths):
        self.imagePaths = imagePaths

    def setLang(self, lang):
        self.lang = lang

    def setOutputDir(self, outputDir):
        self.outputDir = outputDir

    def setPDFParser(self, enabled):
        self.use_pdf2docx_api = enabled

    def resetPageCnt(self):
        self.pageCnt = 0

    def resetTotalPageCnt(self):
        self.totalPageCnt = 0

    def setKeepReference(self, keep_reference):
        self.keep_reference = keep_reference

    def _convert_res_for_draw(self, res):
        """
        将识别结果转换为 draw_structure_result 期望的格式
        将 text_region [x1,y1,x2,y2] 转换为四点格式 [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
        """
        res_converted = []
        for region in res:
            region_copy = copy.deepcopy(region)
            
            # 处理 res 列表中的 text_region
            if isinstance(region_copy.get('res'), list):
                for item in region_copy['res']:
                    if isinstance(item, dict) and 'text_region' in item:
                        text_region = item['text_region']
                        if isinstance(text_region, list) and len(text_region) == 4:
                            # 转换 [x1,y1,x2,y2] 为四点格式
                            x1, y1, x2, y2 = text_region
                            item['text_region'] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            
            res_converted.append(region_copy)
        
        return res_converted

    def ppocrPrecitor(self, imgs, img_name):
        all_res = []
        # processing pages (总页数已在run()开始时预计算)
        for index, img in enumerate(imgs):
            # 注意：去红章逻辑已经在predict_system.py中处理，这里不再需要预处理
            res, time_dict = self.predictors[self.lang](img)

            # save output
            save_structure_res(res, self.outputDir, img_name)
            
            # 转换 res 格式以适配 draw_structure_result
            # draw_structure_result 期望 text_region 是 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] 格式
            res_for_draw = self._convert_res_for_draw(res)
            draw_img = draw_structure_result(img, res_for_draw, self.vis_font_path)
            
            # 规范化路径，确保使用正确的路径分隔符
            img_dir = os.path.normpath(os.path.join(self.outputDir, img_name))
            os.makedirs(img_dir, exist_ok=True)
            img_save_path = os.path.normpath(os.path.join(img_dir, 'show_{}.jpg'.format(index)))
            print('\naaaaaaaa\naaaaaa\n')
            if res != []:
                save_image_with_chinese_path(draw_img, img_save_path)
                print('\n\n[image saved at ]', img_save_path)

            # recovery - 保存每一页的识别结果和对应的图像
            h, w, _ = img.shape
            res = sorted_layout_boxes(res, w)
            # 为每一页的结果添加页码标记
            for region in res:
                region['page_index'] = index
            all_res.append({
                'img': img,
                'res': res,
                'page_index': index
            })
            self.pageCnt += 1
            self.progressBarValue.emit(self.pageCnt)

        if all_res != []:
            try:
                # 规范化输出目录路径
                normalized_output_dir = os.path.normpath(self.outputDir)
                # 把所有页面放在一个 Word 文档里，用分页符分隔
                convert_info_docx_multi_page(all_res, normalized_output_dir, img_name, self.keep_reference)
            except Exception as ex:
                warning_msg = "⚠️ 警告：版面恢复过程中出现错误\n\n文件：{}\n错误信息：{}\n\n但是识别的内容已保存。".format(
                    img_name, str(ex))
                print(warning_msg)
                print("Traceback:\n{}".format(traceback.format_exc()))
        
        print("Predict time : {:.3f}s".format(time_dict['all']))
        print('result save to {}'.format(os.path.normpath(self.outputDir)))

    def run(self):
        self.resetPageCnt()
        self.resetTotalPageCnt()
        try:
            os.makedirs(self.outputDir, exist_ok=True)
            
            # 预先计算总页数，一次性设置进度条范围
            total_pages = 0
            for image_file in self.imagePaths:
                file_ext = os.path.basename(image_file)[-3:].lower()
                if file_ext == 'pdf':
                    try:
                        with fitz.open(image_file) as pdf:
                            if self.use_pdf2docx_api:
                                # 检查是否为扫描版PDF
                                text_count = 0
                                for page_num in range(pdf.page_count):
                                    page = pdf[page_num]
                                    text = page.get_text()
                                    text_count += len(text.strip())
                                if text_count == 0:
                                    # 扫描版PDF，按页数计算
                                    total_pages += pdf.page_count
                                else:
                                    # 文字版PDF，算1个
                                    total_pages += 1
                            else:
                                # OCR模式，按页数计算
                                total_pages += pdf.page_count
                    except:
                        total_pages += 1  # 出错时默认算1页
                else:
                    # 图片文件算1页
                    total_pages += 1
            
            self.totalPageCnt = total_pages
            self.progressBarRange.emit(self.totalPageCnt)
            
            for i, image_file in enumerate(self.imagePaths):
                if not self.loopFlag:
                    break
                # using use_pdf2docx_api for PDF parsing
                if self.use_pdf2docx_api \
                    and os.path.basename(image_file)[-3:].lower() == 'pdf':
                    print(
                        '===============using use_pdf2docx_api===============')
                    img_name = os.path.basename(image_file).split('.')[0]
                    docx_file = os.path.join(self.outputDir,
                                             '{}.docx'.format(img_name))
                    try:
                        # 处理文件锁定问题
                        docx_file = get_available_docx_path(docx_file)
                        
                        # 检查PDF是否为扫描版本（通过检查文本内容）
                        try:
                            with fitz.open(image_file) as pdf:
                                text_count = 0
                                for page_num in range(pdf.page_count):
                                    page = pdf[page_num]
                                    text = page.get_text()
                                    text_count += len(text.strip())
                                
                                if text_count == 0:
                                    # 扫描版PDF，自动使用OCR转换，不弹窗
                                    print("检测到扫描版PDF，自动使用OCR进行转换: {}".format(image_file))
                                    # 读取PDF为图片
                                    try:
                                        imgs = []
                                        with fitz.open(image_file) as pdf_doc:
                                            for page_num in range(pdf_doc.page_count):
                                                page = pdf_doc[page_num]
                                                # 设置缩放因子以获得更高质量的图像
                                                zoom = 2.0
                                                mat = fitz.Matrix(zoom, zoom)
                                                pix = page.get_pixmap(matrix=mat)
                                                img_data = pix.tobytes("png")
                                                import numpy as np
                                                nparr = np.frombuffer(img_data, np.uint8)
                                                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                                imgs.append(img)
                                        
                                        # 使用OCR处理（总页数已在run()开始时预计算）
                                        self.ppocrPrecitor(imgs, img_name)
                                    except Exception as ocr_err:
                                        print(f"OCR转换出错: {ocr_err}")
                                        print(traceback.format_exc())
                                    continue
                        except Exception as e:
                            print(f"Error checking PDF content: {e}")
                        
                        cv = Converter(image_file)
                        cv.convert(docx_file)
                        cv.close()
                        print('docx save to {}'.format(docx_file))
                        self.pageCnt += 1
                        self.progressBarValue.emit(self.pageCnt)
                    except PermissionError as pex:
                        error_msg = "File permission error for: {}\nThe file may be opened in another program (e.g., Word, Explorer).\nTips: Please close the file and try again, or the program will auto-rename it.\n\nError: {}\nTraceback:\n{}".format(
                            docx_file, str(pex), traceback.format_exc())
                        print(error_msg)
                        self.exceptedsignal.emit(error_msg)
                        return
                    except Exception as ex:
                        error_msg = "Error processing PDF file: {}\nError: {}\nTraceback:\n{}".format(
                            image_file, str(ex), traceback.format_exc())
                        print(error_msg)
                        self.exceptedsignal.emit(error_msg)
                        return
                else:
                    # using PPOCR for PDF/Image parsing
                    imgs = readImage(image_file)
                    if len(imgs) == 0:
                        continue
                    img_name = os.path.basename(image_file).split('.')[0]
                    os.makedirs(
                        os.path.join(self.outputDir, img_name), exist_ok=True)
                    self.ppocrPrecitor(imgs, img_name)
                # file processed
            self.endsignal.emit()
            # self.exec()
        except Exception as e:
            error_msg = "Fatal error occurred:\nError: {}\nTraceback:\n{}".format(
                str(e), traceback.format_exc())
            print(error_msg)
            self.exceptedsignal.emit(error_msg)  # 将异常发送给UI进程


class APP_Image2Doc(QWidget):
    def __init__(self):
        super().__init__()
        # self.setFixedHeight(100)
        # self.setFixedWidth(520)

        # settings
        self.imagePaths = []
        # self.screenShotWg = ScreenShotWidget()
        self.screenShot = None
        self.save_pdf = False
        self.output_dir = None
        self.vis_font_path = os.path.join(root, "doc", "fonts", "simfang.ttf")
        self.use_pdf2docx_api = False
        self.keep_reference = True  # 是否保留签批区

        # ProgressBar
        self.pb = QProgressBar()
        self.pb.setRange(0, 100)
        self.pb.setValue(0)

        # 初始化界面
        self.setupUi()

        # 下载模型
        # self.downloadModels(URLs_EN)
        # self.downloadModels(URLs_CN)

        # 初始化模型
        predictors = {
            # 'EN': self.initPredictor('EN'),
            'CN': self.initPredictor('CN'),
        }

        # 设置工作进程
        self._thread = Worker(predictors, self.save_pdf, self.vis_font_path,
                              self.use_pdf2docx_api, self.keep_reference)
        self._thread.progressBarValue.connect(
            self.handleProgressBarUpdateSingal)
        self._thread.endsignal.connect(self.handleEndsignalSignal)
        # self._thread.finished.connect(QObject.deleteLater)
        self._thread.progressBarRange.connect(self.handleProgressBarRangeSingal)
        self._thread.exceptedsignal.connect(self.handleThreadException)
        self._thread.warningsignal.connect(self.handleWarningSignal)  # 连接警告信号
        self.time_start = 0  # save start time
        
        # 创建计时器用于更新duration
        self.duration_timer = QTimer()
        self.duration_timer.timeout.connect(self.updateDuration)

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.setWindowTitle(__APPNAME__ + " " + __VERSION__)

        layout = QGridLayout()

        self.openFileButton = QPushButton("选择文件")
        self.openFileButton.setIcon(QIcon(QPixmap("./icons/folder-plus.png")))
        layout.addWidget(self.openFileButton, 0, 0, 1, 1)
        self.openFileButton.clicked.connect(self.handleOpenFileSignal)

        # screenShotButton = QPushButton("截图识别")
        # layout.addWidget(screenShotButton, 0, 1, 1, 1)
        # screenShotButton.clicked.connect(self.screenShotSlot)
        # screenShotButton.setEnabled(False) # temporarily disenble

        self.startCNButton = QPushButton("中文转换")
        self.startCNButton.setIcon(QIcon(QPixmap("./icons/chinese.png")))
        layout.addWidget(self.startCNButton, 0, 1, 1, 1)  # 隐藏此按钮
        self.startCNButton.clicked.connect(
            functools.partial(self.handleStartSignal, 'CN', False))

        self.startENButton = QPushButton("英文转换")
        self.startENButton.setIcon(QIcon(QPixmap("./icons/english.png")))
        # layout.addWidget(self.startENButton, 0, 2, 1, 1)  # 隐藏此按钮
        self.startENButton.clicked.connect(
            functools.partial(self.handleStartSignal, 'EN', False))

        self.PDFParserButton = QPushButton('PDF解析', self)
        layout.addWidget(self.PDFParserButton, 0, 3, 1, 1)
        self.PDFParserButton.clicked.connect(
            functools.partial(self.handleStartSignal, 'CN', True))

        self.keepReferenceCheckBox = QCheckBox("保留签批区")
        self.keepReferenceCheckBox.setChecked(True)  # 默认勾选
        layout.addWidget(self.keepReferenceCheckBox, 0, 4, 1, 1)
        self.keepReferenceCheckBox.stateChanged.connect(self.handleKeepReferenceCheckBoxSignal)

        self.showResultButton = QPushButton("显示结果")
        self.showResultButton.setIcon(QIcon(QPixmap("./icons/folder-open.png")))
        layout.addWidget(self.showResultButton, 0, 5, 1, 1)
        self.showResultButton.clicked.connect(self.handleShowResultSignal)

        self.resetButton = QPushButton("重置")
        self.resetButton.setIcon(QIcon(QPixmap("./icons/refresh.png")))
        layout.addWidget(self.resetButton, 0, 6, 1, 1)
        self.resetButton.clicked.connect(self.handleResetSignal)

        # 消息显示区域（使用QTextEdit支持滚动）
        self.messageTextEdit = QTextEdit()
        self.messageTextEdit.setReadOnly(True)
        self.messageTextEdit.setMaximumHeight(120)
        self.messageTextEdit.setMinimumHeight(80)
        self.messageTextEdit.setStyleSheet("QTextEdit { background-color: #f5f5f5; border: 1px solid #ccc; }")
        self.messageTextEdit.setPlaceholderText("已选择文件: 无")
        layout.addWidget(self.messageTextEdit, 1, 0, 1, 7)

        # ProgressBar
        layout.addWidget(self.pb, 2, 0, 1, 7)
        # 时间标签
        self.durationLabel = QLabel("Duration: 00:00:00")
        layout.addWidget(self.durationLabel, 3, 0, 1, 3)
        self.timeEstLabel = QLabel(("Time Left: --"))
        layout.addWidget(self.timeEstLabel, 3, 3, 1, 4)

        self.setLayout(layout)

    def downloadModels(self, URLs):
        # using custom model
        tar_file_name_list = [
            'inference.pdiparams', 'inference.pdiparams.info',
            'inference.pdmodel', 'model.pdiparams', 'model.pdiparams.info',
            'model.pdmodel'
        ]
        model_path = os.path.join(root, 'inference')
        os.makedirs(model_path, exist_ok=True)

        # download and unzip models
        for name in URLs.keys():
            url = URLs[name]
            print("Try downloading file: {}".format(url))
            tarname = url.split('/')[-1]
            tarpath = os.path.join(model_path, tarname)
            if os.path.exists(tarpath):
                print("File have already exist. skip")
            else:
                try:
                    download_with_progressbar(url, tarpath)
                except Exception as e:
                    print(
                        "Error occurred when downloading file, error message:")
                    print(e)

            # unzip model tar
            try:
                with tarfile.open(tarpath, 'r') as tarObj:
                    storage_dir = os.path.join(model_path, name)
                    os.makedirs(storage_dir, exist_ok=True)
                    for member in tarObj.getmembers():
                        filename = None
                        for tar_file_name in tar_file_name_list:
                            if tar_file_name in member.name:
                                filename = tar_file_name
                        if filename is None:
                            continue
                        file = tarObj.extractfile(member)
                        with open(os.path.join(storage_dir, filename),
                                  'wb') as f:
                            f.write(file.read())
            except Exception as e:
                print("Error occurred when unziping file, error message:")
                print(e)

    def initPredictor(self, lang='CN'):
        # init predictor args
        args = parse_args()
        args.use_gpu=False
        if hasattr(args, "use_tensorrt"):
            args.use_tensorrt = False
        if hasattr(args, "precision"):
            args.precision = "fp32"

            # CPU 性能参数（可选）
        if hasattr(args, "cpu_threads"):
            args.cpu_threads = max(2, min(8, os.cpu_count() or 4))

            # MKLDNN（Windows wheel 不一定支持；不确定就别开）
        if hasattr(args, "enable_mkldnn"):
            args.enable_mkldnn = False

        args.table_max_len = 488
        args.ocr = True
        args.recovery = True
        args.save_pdf = self.save_pdf
        args.table_char_dict_path = os.path.join(root, "ppocr", "utils", "dict",
                                                 "table_structure_dict.txt")
        # print('\nlang:::',lang)
        if lang == 'EN':
            args.det_model_dir = os.path.join(
                root,  # 此处从这里找到模型存放位置
                "inference",
                "en_PP-OCRv3_det_infer")
            args.rec_model_dir = os.path.join(root, "inference",
                                              "en_PP-OCRv3_rec_infer")
            args.table_model_dir = os.path.join(
                root, "inference", "en_ppstructure_mobile_v2.0_SLANet_infer")
            args.output = os.path.join(root, "output")  # 结果保存路径
            args.layout_model_dir = os.path.join(
                root, "inference", "picodet_lcnet_x1_0_fgd_layout_infer")
            lang_dict = DICT_EN
        elif lang == 'CN':
            args.det_model_dir = os.path.join(
                root,  # 此处从这里找到模型存放位置
                "inference",
                "cn_PP-OCRv3_det_infer")
            args.rec_model_dir = os.path.join(root, "inference",
                                              "cn_PP-OCRv3_rec_infer")
            args.table_model_dir = os.path.join(
                root, "inference", "cn_ppstructure_mobile_v2.0_SLANet_infer")
            args.output = os.path.join(root, "output")  # 结果保存路径
            # args.layout_model_dir = os.path.join(
            #     root, "inference", "picodet_lcnet_x1_0_fgd_layout_cdla_infer")
            args.layout_model_dir = r'/data/tensorflow/kath/Service/pdf2word/models/pdf2wordlayout'
            args.layout_dict_path = os.path.join(
                root, "inference", "pdf2wordlayout")
            args.layout_score_threshold=0.6
            print('[layout model dir]', args.layout_model_dir)
            lang_dict = DICT_CN
        else:
            raise ValueError("Unsupported language")
        args.rec_char_dict_path = os.path.join(root, "ppocr", "utils",
                                               lang_dict['rec_char_dict_path'])
        args.layout_dict_path = os.path.join(root, "ppocr", "utils", "dict",
                                             "layout_dict",
                                             lang_dict['layout_dict_path'])
        # init predictor
        return StructureSystem(args)

    def handleOpenFileSignal(self):
        '''
        可以多选图像文件
        '''
        selectedFiles = QFileDialog.getOpenFileNames(
            self, "多文件选择", "/", "图片文件 (*.png *.jpeg *.jpg *.bmp *.pdf)")[0]
        if len(selectedFiles) > 0:
            self.imagePaths = selectedFiles
            self.screenShot = None  # discard screenshot temp image
            self.pb.setValue(0)
            
            # 显示已选择的文件（每个文件换行显示）
            file_list = [os.path.basename(f) for f in selectedFiles]
            file_display = "已选择文件:\n" + "\n".join(file_list)
            self.messageTextEdit.setText(file_display)
            print(f"Selected files: {selectedFiles}")

    # def screenShotSlot(self):
    #     '''
    #     选定图像文件和截图的转换过程只能同时进行一个
    #     截图只能同时转换一个
    #     '''
    #     self.screenShotWg.start()
    #     if self.screenShotWg.captureImage:
    #         self.screenShot = self.screenShotWg.captureImage
    #         self.imagePaths.clear() # discard openfile temp list
    #         self.pb.setRange(0, 1)
    #         self.pb.setValue(0)

    def handleStartSignal(self, lang='EN', pdfParser=False):
        # 每次点击转换按钮，清空之前的选择和文字（如果没有已选文件）
        if len(self.imagePaths) == 0 and not self.screenShot:
            self.messageTextEdit.clear()
            self.pb.setValue(0)
            QMessageBox.warning(self, u'Information', "请选择要识别的文件或截图")
            return
        
        if self.screenShot:  # for screenShot
            img_name = 'screenshot_' + time.strftime("%Y%m%d%H%M%S",
                                                     time.localtime())
            image = QImageToCvMat(self.screenShot)
            self.predictAndSave(image, img_name, lang)
            # update Progress Bar
            self.pb.setValue(1)
            QMessageBox.information(self, u'Information', "文档提取完成")
        elif len(self.imagePaths) > 0:  # for image file selection
            # Must set image path list and language before start
            # 规范化输出目录路径，统一使用反斜杠
            self.output_dir = os.path.normpath(os.path.join(
                os.path.dirname(self.imagePaths[0]),
                "output"))
            
            self._thread.setOutputDir(self.output_dir)
            self._thread.setImagePath(self.imagePaths)
            self._thread.setLang(lang)
            self._thread.setPDFParser(pdfParser)
            # 根据勾选框设置保留签批区标志
            self.keep_reference = self.keepReferenceCheckBox.isChecked()
            self._thread.setKeepReference(self.keep_reference)
            # disenble buttons
            self.openFileButton.setEnabled(False)
            self.startCNButton.setEnabled(False)
            self.startENButton.setEnabled(False)
            self.PDFParserButton.setEnabled(False)
            # 启动工作进程
            self._thread.start()
            self.time_start = time.time()  # log start time
            self.duration_timer.start(1000)  # 每1000ms(1秒)更新一次duration
            # 在消息区域显示开始转换
            self.appendMessage("开始转换...")

    def handleShowResultSignal(self):
        if self.output_dir is None:
            return
        if os.path.exists(self.output_dir):
            if platform.system() == 'Windows':
                os.startfile(self.output_dir)
            else:
                os.system('open ' + os.path.normpath(self.output_dir))
        else:
            QMessageBox.information(self, u'Information', "输出文件不存在")

    def handleProgressBarUpdateSingal(self, i):
        self.pb.setValue(i)
        # calculate time left of recognition
        total_pages = self.pb.maximum()
        current_time = time.time()
        
        # 确保total_pages有效
        if total_pages <= 0:
            self.timeEstLabel.setText("Time Left: --")
            self.updateDuration()
            return
        
        # 计算剩余时间
        remaining_pages = total_pages - i
        
        if i > 0:
            # 使用从开始到现在的总时间来计算平均速度（更稳定）
            elapsed_since_start = current_time - self.time_start
            
            if elapsed_since_start > 0:
                # 每页平均耗时 = 总耗时 / 已完成页数
                avg_time_per_page = elapsed_since_start / i
                
                # 计算剩余时间
                estimated_seconds = int(avg_time_per_page * remaining_pages)
                
                # 格式化时间显示
                if estimated_seconds >= 0:
                    hours = estimated_seconds // 3600
                    minutes = (estimated_seconds % 3600) // 60
                    seconds = estimated_seconds % 60
                    time_left = f"{hours}:{minutes:02d}:{seconds:02d}"
                else:
                    time_left = "0:00:00"
            else:
                time_left = "--"
        else:
            time_left = "--"
        
        self.timeEstLabel.setText(f"Time Left: {time_left}")
        self.updateDuration()

    def updateDuration(self):
        """更新已用时间显示"""
        if self.time_start > 0:
            elapsed_time = time.time() - self.time_start
            duration = datetime.timedelta(seconds=int(elapsed_time))
            self.durationLabel.setText(f"Duration: {duration}")

    def handleProgressBarRangeSingal(self, max):
        self.pb.setRange(0, max)

    def appendMessage(self, msg):
        """在消息区域追加一行消息"""
        current_text = self.messageTextEdit.toPlainText()
        if current_text:
            self.messageTextEdit.setText(current_text + "\n" + msg)
        else:
            self.messageTextEdit.setText(msg)
        # 滚动到底部
        self.messageTextEdit.verticalScrollBar().setValue(
            self.messageTextEdit.verticalScrollBar().maximum())

    def handleEndsignalSignal(self):
        # enble buttons
        self.duration_timer.stop()  # 停止计时
        self.openFileButton.setEnabled(True)
        self.startCNButton.setEnabled(True)
        self.startENButton.setEnabled(True)
        self.PDFParserButton.setEnabled(True)
        self.resetButton.setEnabled(True)
        
        # 清除已识别完的文件
        self.imagePaths = []
        self.screenShot = None
        self.pb.setValue(0)
        self.messageTextEdit.clear()
        self.messageTextEdit.setPlaceholderText("已选择文件: 无")
        
        # 在消息区域显示转换结束
        self.appendMessage("转换结束，文件列表已清空")
        QMessageBox.information(self, u'Information', "转换结束")

    def handleKeepReferenceCheckBoxSignal(self):
        """
        处理保留签批区勾选框的信号
        """
        self.keep_reference = self.keepReferenceCheckBox.isChecked()
        if self.keep_reference:
            QMessageBox.information(self, u'Information', "已启用保留签批区")
        else:
            QMessageBox.information(self, u'Information', "已禁用保留签批区")

    def handleWarningSignal(self, warning_msg):
        """
        处理来自Worker线程的警告信号
        """
        self.appendMessage("⚠️ " + warning_msg)
        QMessageBox.warning(self, u'警告', warning_msg)

    def handleCBChangeSignal(self):
        self._thread.setPDFParser(self.checkBox.isChecked())

    def handleThreadException(self, message):
        self.duration_timer.stop()  # 停止计时
        self._thread.quit()
        # 创建一个详细的错误对话框，显示完整的错误追溯
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle('Error')
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText('处理过程中发生错误')
        msg_box.setDetailedText(message)  # 在详细信息中显示完整的traceback
        msg_box.setMinimumWidth(700)
        msg_box.setMinimumHeight(500)
        msg_box.exec()
        # 恢复按钮状态
        self.openFileButton.setEnabled(True)
        self.startCNButton.setEnabled(True)
        self.startENButton.setEnabled(True)
        self.PDFParserButton.setEnabled(True)
        self.resetButton.setEnabled(True)

    def handleResetSignal(self):
        '''
        重置应用状态，允许新的识别任务
        '''
        # 检查线程是否还在运行
        if self._thread.isRunning():
            self._thread.loopFlag = False  # 停止线程
            self._thread.wait()  # 等待线程完全停止
        
        # 重置状态变量
        self.imagePaths = []
        self.screenShot = None
        self.output_dir = None
        self.pb.setValue(0)
        self.pb.setRange(0, 100)
        self.timeEstLabel.setText("Time Left: --")
        self.durationLabel.setText("Duration: 00:00:00")
        self.duration_timer.stop()  # 停止计时器
        self.time_start = 0
        self.messageTextEdit.clear()
        self.messageTextEdit.setPlaceholderText("已选择文件: 无")
        
        # 重新创建Worker线程
        self._thread = Worker({
            'CN': self.initPredictor('CN'),
        }, self.save_pdf, self.vis_font_path, self.use_pdf2docx_api, self.keep_reference)
        self._thread.progressBarValue.connect(self.handleProgressBarUpdateSingal)
        self._thread.endsignal.connect(self.handleEndsignalSignal)
        self._thread.progressBarRange.connect(self.handleProgressBarRangeSingal)
        self._thread.exceptedsignal.connect(self.handleThreadException)
        self._thread.warningsignal.connect(self.handleWarningSignal)  # 连接警告信号
        
        # 启用所有按钮
        self.openFileButton.setEnabled(True)
        self.startCNButton.setEnabled(True)
        self.startENButton.setEnabled(True)
        self.PDFParserButton.setEnabled(True)
        self.resetButton.setEnabled(True)
        
        QMessageBox.information(self, u'Information', "应用已重置，可以开始新的识别")

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
def main():
    app = QApplication(sys.argv)

    window = APP_Image2Doc()  # 创建对象
    window.show()  # 全屏显示窗口

    QApplication.processEvents()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
