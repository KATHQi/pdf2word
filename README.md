# PDF2Word / OCR Extraction Service

基于 PaddleOCR / ppstructure 的 PDF 转 Word 以及特定关键信息提取服务。

## 特性

- 自动区分文字版可直接提取的 PDF 和扫描版 PDF。
- 对扫描版 PDF 调用 PaddleOCR 进行布局分析、文本检测、识别和表格结构恢复，转换为可编辑的 Word (.docx)。
- 特殊表单提取 API，用于合同和申请表的批量结构化抽取，返回 Excel。

## 安装依赖

`ash
pip install -r requirements.txt
`

## 运行服务

`ash
cd ppstructure/pdf2word
python web_server.py
`
服务将在 http://localhost:8005 启动。

## 注意事项
模型文件(inference等)及测试数据(web_output/, web_uploads/)已被忽略，请自行下载相应的 PaddleOCR/PP-Structure 模型并放到对应路径下。