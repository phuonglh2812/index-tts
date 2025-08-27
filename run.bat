@echo off
:: Kích hoạt môi trường "index-tts" từ thư mục Anaconda bạn đã cài ở E:\Anaconda
CALL "E:\Anaconda\Scripts\activate.bat" index-tts

:: Chạy script Python (đặt cùng thư mục hoặc chỉnh đường dẫn nếu khác)
python gui2.py

:: Giữ cửa sổ mở nếu có lỗi
pause
