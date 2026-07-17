# -*- coding: utf-8 -*-
# Command-line version of the original Qt GUI app.
# Keep the same core pipeline:
#  - optional pdf2docx api for parsing PDF
#  - otherwise PP-Structure: layout + table + OCR + recovery to docx

import os
import sys
import time
import datetime
import tarfile
import argparse

import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image
from pdf2docx.converter import Converter

# --- project path wiring (keep consistent with your repo layout) ---
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(FILE_DIR, "../../"))
sys.path.append(FILE_DIR)
sys.path.insert(0, ROOT_DIR)

from ppstructure.predict_system import StructureSystem, save_structure_res
from ppstructure.utility import parse_args
from ppocr.utils.network import download_with_progressbar
from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes, convert_info_docx


URLs_EN = {
    "en_PP-OCRv3_det_infer":
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar",
    "en_PP-OCRv3_rec_infer":
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar",
    "en_ppstructure_mobile_v2.0_SLANet_infer":
    "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
    "picodet_lcnet_x1_0_fgd_layout_infer":
    "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_infer.tar",
}
DICT_EN = {
    "rec_char_dict_path": "en_dict.txt",
    "layout_dict_path": "layout_publaynet_dict.txt",
}

URLs_CN = {
    "cn_PP-OCRv3_det_infer":
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar",
    "cn_PP-OCRv3_rec_infer":
    "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
    "cn_ppstructure_mobile_v2.0_SLANet_infer":
    "https://paddleocr.bj.bcebos.com/ppstructure/models/slanet/en_ppstructure_mobile_v2.0_SLANet_infer.tar",
    "picodet_lcnet_x1_0_fgd_layout_cdla_infer":
    "https://paddleocr.bj.bcebos.com/ppstructure/models/layout/picodet_lcnet_x1_0_fgd_layout_cdla_infer.tar",
}
DICT_CN = {
    "rec_char_dict_path": "ppocr_keys_v1.txt",
    "layout_dict_path": "layout_cdla_dict.txt",
}


def _ppstructure_default_args():
    """
    parse_args() in PaddleOCR usually consumes sys.argv, which conflicts with our CLI args.
    We obtain a "default args" object safely.

    - If parse_args supports parse_args(args=[]), use it.
    - Otherwise temporarily patch sys.argv to avoid consuming our CLI options.
    """
    try:
        return parse_args(args=[])
    except TypeError:
        old_argv = sys.argv
        sys.argv = [old_argv[0]]
        try:
            return parse_args()
        finally:
            sys.argv = old_argv


def read_image_or_pdf(image_file) -> list:
    """Return a list of BGR images (np.ndarray). For PDF: one image per page."""
    ext = os.path.splitext(image_file)[1].lower()
    if ext == ".pdf":
        imgs = []
        with fitz.open(image_file) as pdf:
            for pg in range(pdf.page_count):
                page = pdf[pg]
                mat = fitz.Matrix(2, 2)
                pm = page.get_pixmap(matrix=mat, alpha=False)

                # if width or height > 2000 pixels, don't enlarge the image
                if pm.width > 2000 or pm.height > 2000:
                    pm = page.get_pixmap(matrix=fitz.Matrix(1, 1), alpha=False)

                img = Image.frombytes("RGB", [pm.width, pm.height], pm.samples)
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                imgs.append(img)
        return imgs

    img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    return [img] if img is not None else []


def count_total_pages(paths):
    """Best-effort: count total pages to support ETA."""
    total = 0
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".pdf":
            try:
                with fitz.open(p) as pdf:
                    total += pdf.page_count
            except Exception:
                total += 1
        else:
            total += 1
    return total


def download_models(urls: dict, root_dir: str):
    tar_file_name_list = [
        "inference.pdiparams", "inference.pdiparams.info", "inference.pdmodel",
        "model.pdiparams", "model.pdiparams.info", "model.pdmodel"
    ]
    model_path = os.path.join(root_dir, "inference")
    os.makedirs(model_path, exist_ok=True)

    for name, url in urls.items():
        print(f"[Model] Try downloading: {url}")
        tarname = url.split("/")[-1]
        tarpath = os.path.join(model_path, tarname)

        if not os.path.exists(tarpath):
            try:
                download_with_progressbar(url, tarpath)
            except Exception as e:
                print("[Model] Download error:", e)
                continue
        else:
            print("[Model] Tar already exists, skip download.")

        # unzip model tar
        try:
            with tarfile.open(tarpath, "r") as tar_obj:
                storage_dir = os.path.join(model_path, name)
                os.makedirs(storage_dir, exist_ok=True)
                for member in tar_obj.getmembers():
                    filename = None
                    for keep in tar_file_name_list:
                        if keep in member.name:
                            filename = keep
                            break
                    if filename is None:
                        continue
                    fobj = tar_obj.extractfile(member)
                    if fobj is None:
                        continue
                    with open(os.path.join(storage_dir, filename), "wb") as f:
                        f.write(fobj.read())
        except Exception as e:
            print("[Model] Unzip error:", e)


def init_predictor(lang: str, save_pdf: bool, root_dir: str, vis_font_path: str):
    args = _ppstructure_default_args()

    # keep the same defaults as your GUI version
    args.table_max_len = 488
    args.ocr = True
    args.recovery = True
    args.save_pdf = save_pdf
    args.table_char_dict_path = os.path.join(
        root_dir, "ppocr", "utils", "dict", "table_structure_dict.txt"
    )

    if lang == "EN":
        args.det_model_dir = os.path.join(root_dir, "inference", "en_PP-OCRv3_det_infer")
        args.rec_model_dir = os.path.join(root_dir, "inference", "en_PP-OCRv3_rec_infer")
        args.table_model_dir = os.path.join(root_dir, "inference", "en_ppstructure_mobile_v2.0_SLANet_infer")
        args.layout_model_dir = os.path.join(root_dir, "inference", "picodet_lcnet_x1_0_fgd_layout_infer")
        lang_dict = DICT_EN
    elif lang == "CN":
        args.det_model_dir = os.path.join(root_dir, "inference", "cn_PP-OCRv3_det_infer")
        args.rec_model_dir = os.path.join(root_dir, "inference", "cn_PP-OCRv3_rec_infer")
        args.table_model_dir = os.path.join(root_dir, "inference", "cn_ppstructure_mobile_v2.0_SLANet_infer")
        args.layout_model_dir = os.path.join(root_dir, "inference", "picodet_lcnet_x1_0_fgd_layout_cdla_infer")
        lang_dict = DICT_CN
    else:
        raise ValueError("Unsupported language. Use CN or EN.")

    args.output = os.path.join(root_dir, "output")  # will be overridden by CLI output_dir
    args.rec_char_dict_path = os.path.join(root_dir, "ppocr", "utils", lang_dict["rec_char_dict_path"])
    args.layout_dict_path = os.path.join(
        root_dir, "ppocr", "utils", "dict", "layout_dict", lang_dict["layout_dict_path"]
    )
    args.vis_font_path = vis_font_path  # some versions use it in drawing

    return StructureSystem(args)


def ppocr_predictor(predictor, imgs, img_name, output_dir):
    all_res = []
    last_time_dict = None

    for idx, img in enumerate(imgs):
        res, time_dict = predictor(img)
        last_time_dict = time_dict

        # save intermediate structure results (same as GUI)
        save_structure_res(res, output_dir, img_name)

        # recovery
        h, w, _ = img.shape
        res = sorted_layout_boxes(res, w)
        all_res += res

    if all_res:
        try:
            convert_info_docx(imgs, all_res, output_dir, img_name)
        except Exception as ex:
            print(f"[Recovery] error: image={img_name}, err={ex}")

    if last_time_dict and "all" in last_time_dict:
        print(f"[Predict] time: {last_time_dict['all']:.3f}s")
    print(f"[Output] result saved to: {output_dir}")


def process_files(
    paths, lang, output_dir, use_pdf2docx_api, save_pdf, vis_font_path, no_download
):
    if not paths:
        raise ValueError("No input files provided.")

    # download models if needed
    if not no_download:
        if lang == "CN":
            download_models(URLs_CN, ROOT_DIR)
        else:
            download_models(URLs_EN, ROOT_DIR)

    predictor = init_predictor(lang=lang, save_pdf=save_pdf, root_dir=ROOT_DIR, vis_font_path=vis_font_path)

    os.makedirs(output_dir, exist_ok=True)
    total_pages = count_total_pages(paths)
    done_pages = 0
    t0 = time.time()

    def show_eta():
        nonlocal done_pages
        elapsed = time.time() - t0
        avg = elapsed / max(done_pages, 1)
        left = avg * max(total_pages - done_pages, 0)
        eta = str(datetime.timedelta(seconds=int(left)))
        print(f"[Progress] {done_pages}/{total_pages} pages | ETA {eta}", end="\r", flush=True)

    for image_file in paths:
        ext = os.path.splitext(image_file)[1].lower()
        base = os.path.basename(image_file).rsplit(".", 1)[0]

        if use_pdf2docx_api and ext == ".pdf":
            # pdf2docx path
            docx_file = os.path.join(output_dir, f"{base}.docx")
            print(f"\n[PDF2DOCX] {image_file} -> {docx_file}")
            cv = Converter(image_file)
            cv.convert(docx_file)
            cv.close()
            done_pages += 1
            show_eta()
            continue

        # PP-Structure path
        imgs = read_image_or_pdf(image_file)
        if not imgs:
            print(f"\n[Warn] Cannot read: {image_file}")
            continue

        # mimic GUI behavior: create subfolder output_dir/base
        os.makedirs(os.path.join(output_dir, base), exist_ok=True)

        print(f"\n[PP-Structure] processing: {image_file} (pages={len(imgs)})")
        ppocr_predictor(predictor, imgs, base, output_dir)

        done_pages += len(imgs)
        show_eta()

    print("\n[Done] All files processed.")


def build_parser():
    p = argparse.ArgumentParser(
        description="pdf2word CLI (PaddleOCR PP-Structure / optional pdf2docx)"
    )
    p.add_argument(
        "inputs", nargs="+",
        help="Input files: *.png *.jpg *.jpeg *.bmp *.pdf (multiple allowed)"
    )
    p.add_argument(
        "-l", "--lang", default="CN", choices=["CN", "EN"],
        help="OCR language pipeline (CN or EN)"
    )
    p.add_argument(
        "-o", "--output", default=None,
        help="Output directory. Default: <dir-of-first-input>/output"
    )
    p.add_argument(
        "--pdf-parser", action="store_true",
        help="Use pdf2docx API for PDFs (skip PaddleOCR layout recovery)"
    )
    p.add_argument(
        "--save-pdf", action="store_true",
        help="Forward to PaddleOCR args.save_pdf (keep for compatibility)"
    )
    p.add_argument(
        "--vis-font-path", default=os.path.join(ROOT_DIR, "doc", "fonts", "simfang.ttf"),
        help="Font path used by some visualization / recovery components"
    )
    p.add_argument(
        "--no-download", action="store_true",
        help="Skip model downloading (assume inference/ already exists)"
    )
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    # default output: same as GUI (output under the directory of first file)
    if args.output is None:
        first_dir = os.path.dirname(os.path.abspath(args.inputs[0]))
        output_dir = os.path.join(first_dir, "output")
    else:
        output_dir = os.path.abspath(args.output)

    process_files(
        paths=args.inputs,
        lang=args.lang,
        output_dir=output_dir,
        use_pdf2docx_api=args.pdf_parser,
        save_pdf=args.save_pdf,
        vis_font_path=args.vis_font_path,
        no_download=args.no_download,
    )


if __name__ == "__main__":
    main()
