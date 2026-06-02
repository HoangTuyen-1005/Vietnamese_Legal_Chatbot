import os
import time
import random
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote, urljoin
from dotenv import load_dotenv

load_dotenv()
# ================= CẤU HÌNH HỆ THỐNG =================

RAW_DIR = os.path.join("data_pipeline", "data", "raw")
CHUDE_FILE = os.path.join("data_pipeline", "ChuDe.txt")
BASE_URL = "https://thuvienphapluat.vn"

# TÀI KHOẢN ĐĂNG NHẬP (Để tự động lấy Cookie mới)
TVPL_USERNAME = os.getenv("TVPL_USERNAME")
TVPL_PASSWORD = os.getenv("TVPL_PASSWORD")

# Đảm bảo thư mục tồn tại
os.makedirs(RAW_DIR, exist_ok=True)

# Khởi tạo một phiên (Session)
session = requests.Session()

# HEADER TỐI GIẢN, SẠCH SẼ
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36 Edg/148.0.0.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
})

# ================= HÀM TỰ ĐỘNG ĐĂNG NHẬP =================
def auto_login():
    """Hàm tự động đăng nhập để lấy Cookie tươi mới mỗi lần chạy ETL"""
    print("\n========================================")
    print("Đang tiến hành tự động đăng nhập lấy Cookie...")
    login_url = "https://thuvienphapluat.vn/page/ajaxcontroler.aspx"
    
    # Payload giả lập gửi lên API đăng nhập
    payload = {
        'l_txtUser': TVPL_USERNAME,
        'l_txtPass': TVPL_PASSWORD,
        'action': 'Login'
    }
    
    # Cập nhật Referer giả lập trang chủ
    session.headers.update({'Referer': 'https://thuvienphapluat.vn/'})
    
    try:
        response = session.post(login_url, data=payload, timeout=15)
        response.raise_for_status()
        
        # TVPL thường trả về chữ <ok> hoặc chuỗi rỗng nếu đăng nhập thành công
        if "<ok>" in response.text or response.text.strip() == "":
            print("[Thành công] Đã đăng nhập và lấy được Cookie mới!")
            return True
        else:
            print(f"[Thất bại] Sai tài khoản/mật khẩu hoặc web thay đổi cơ chế. Trả về: {response.text[:100]}")
            return False
            
    except Exception as e:
        print(f"[Lỗi Hệ Thống] Không thể kết nối để đăng nhập: {e}")
        return False

# ================= CÁC HÀM CRAWLER =================
def read_keywords(filepath):
    if not os.path.exists(filepath):
        print(f"Không tìm thấy file {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_document_links(keyword):
    """
    Logic tìm kiếm:
    Bước 1: Tìm VBHN (type=40)
    Bước 2: Kiểm tra Tiêu đề kết quả đầu tiên
    Bước 3a: Nếu Hợp lệ -> Trả về link tải
    Bước 3b: Nếu sai tên (Nghị định, Thông tư...) -> Kích hoạt Fallback
    Bước 4: Đổi bộ lọc sang Luật/Pháp lệnh (type=10) để lấy bản gốc.
    """
    encoded_keyword = quote(keyword)
    
    print(f"\n----------------------------------------")
    print(f"Đang tìm kiếm chủ đề: {keyword}")
    
    try:
        # [BƯỚC 1] TÌM KIẾM VĂN BẢN HỢP NHẤT (type=40)
        search_url_vbhn = f"http://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword={encoded_keyword}&area=0&type=40&status=0&lan=1&org=0&signer=0&match=True&sort=1"
        response = session.get(search_url_vbhn, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        fallback_needed = True 
        
        for a_tag in soup.find_all('a', href=True):
            raw_href = a_tag['href']
            title = (a_tag.get('title', '') or a_tag.text).strip().lower()
            clean_href = raw_href.split('?')[0] 
            
            if '/van-ban/' in clean_href and ('VBHN' in clean_href or 'Van-ban-hop-nhat' in clean_href):
                
                # [BƯỚC 2] KIỂM TRA TIÊU ĐỀ KẾT QUẢ ĐẦU TIÊN
                if "nghị định" in title or "thông tư" in title or "quyết định" in title:
                    print(f"-> [Phát hiện rác] Web trả về VBHN sai loại (Hướng dẫn): {a_tag.text.strip()[:60]}...")
                    fallback_needed = True
                    break 
                
                # [BƯỚC 3a] HỢP LỆ -> TẢI FILE
                full_link = urljoin(BASE_URL, clean_href)
                print(f"-> [Chuẩn Hợp Nhất] Đã chọn VBHN: {full_link}")
                return [full_link] 
                
        # [BƯỚC 3b & BƯỚC 4] KÍCH HOẠT FALLBACK
        if fallback_needed:
            print(f"-> [Fallback kích hoạt] Đổi bộ lọc sang tìm Luật/Pháp lệnh gốc hiện hành...")
            search_url_goc = f"http://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword={encoded_keyword}&area=0&type=10&status=0&lan=1&org=0&signer=0&match=True&sort=1"
            
            response_goc = session.get(search_url_goc, timeout=15)
            response_goc.raise_for_status()
            soup_goc = BeautifulSoup(response_goc.content, 'html.parser')
            
            for a_tag in soup_goc.find_all('a', href=True):
                raw_href = a_tag['href']
                clean_href = raw_href.split('?')[0] 
                
                if '/van-ban/' in clean_href:
                    full_link = urljoin(BASE_URL, clean_href)
                    print(f"-> [Thành công] Đã bắt được Luật gốc: {full_link}")
                    return [full_link]
                    
        print(f"-> [Thất bại] Không tìm thấy bất kỳ văn bản nào cho '{keyword}'")
        return []
        
    except Exception as e:
        print(f"Lỗi khi tìm kiếm {keyword}: {e}")
        return []

def download_document(doc_url, keyword):
    try:
        response = session.get(doc_url, timeout=15)
        response.raise_for_status()
        
        match = re.search(r"__urldl\s*=\s*['\"](/documents/download\.aspx.*?)['\"]", response.text)
        if not match:
            match = re.search(r"__urldl\s*=\s*['\"](https?://files\.thuvienphapluat\.vn.*?)['\"]", response.text)
            
        if not match:
            print(f"   [Lỗi] Không tìm thấy mã ẩn tải file. (Tài khoản của bạn có thể không đủ quyền tải).")
            return
            
        href = match.group(1)
        
        # ================= THAY ĐỔI ĐỂ TẢI FILE .DOCX =================
        # Ép tham số docx rỗng ban đầu chuyển thành docx=1 để tải bản Docx
        if "docx=" in href:
            href = href.replace("docx=", "docx=1")
            
        download_url = urljoin(BASE_URL, href)

        session.headers.update({'Referer': doc_url})

        file_response = session.get(download_url, timeout=30)
        file_response.raise_for_status()

        content_type = file_response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print(f"   [Lỗi] Máy chủ yêu cầu xác thực. Cookie có thể đã hết hạn!")
            return
        
        ext = ".docx" if "docx=1" in download_url else ".doc"
        if ".pdf" in download_url:
            ext = ".pdf"
            
        doc_name = doc_url.split('/')[-1].replace('.aspx', '')
        filename = f"{doc_name}{ext}"

        filepath = os.path.join(RAW_DIR, filename)
        
        with open(filepath, 'wb') as f:
            f.write(file_response.content)
            
        print(f"   [Thành công] Đã tải về data/raw: {filename}")
        
        time.sleep(random.uniform(3.5, 6.0)) 
        
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 403:
             print(f"   [Lỗi 403] Tường lửa chặn tải file. Vui lòng lấy lại Cookie mới.")
        else:
             print(f"   [Lỗi] HTTP Error: {err}")
    except Exception as e:
        print(f"   [Lỗi] Lỗi kết nối khi tải: {e}")

def main():
    # 1. Khởi chạy đăng nhập trước khi làm mọi thứ
    if not auto_login():
        print("Dừng quá trình do không thể đăng nhập lấy Cookie.")
        return

    # 2. Đọc file
    keywords = read_keywords(CHUDE_FILE)
    if not keywords:
        print(f"Danh sách chủ đề trống. Vui lòng thêm từ khóa vào file {CHUDE_FILE}.")
        return

    # 3. Tiến hành tìm kiếm & Tải file
    for keyword in keywords:
        doc_links = get_document_links(keyword)
        if doc_links:
            download_document(doc_links[0], keyword)

if __name__ == "__main__":
    main()