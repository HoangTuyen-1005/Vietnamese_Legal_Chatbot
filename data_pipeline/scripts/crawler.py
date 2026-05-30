import os
import time
import random
import requests
import re
from bs4 import BeautifulSoup
from urllib.parse import quote, urljoin

# Cấu hình thư mục
RAW_DIR = os.path.join("data", "raw")
CHUDE_FILE = "ChuDe.txt"

os.makedirs(RAW_DIR, exist_ok=True)

# Khởi tạo một phiên (Session)
session = requests.Session()

# 1. XỬ LÝ COOKIE THÔNG MINH
# Thay vì nhồi thẳng vào Header, ta chia nhỏ Cookie để thư viện tự quản lý theo tên miền
COOKIE_STRING = 'G_ENABLED_IDPS=google; _ga_SH5R2R5VQ1=GS2.1.s1752975774$o1$g1$t1752975795$j39$l0$h0; orig_aid=2qt2291aqa5vfndb.1773937345.des; __uidac=0169bc22c4d6436ccffcece1ac6c23d2; _pubcid=381039bc-d564-4836-93b2-aa82a190e208; _pubcid_cst=znv0HA%3D%3D; __RC=5; __R=3; _ga_J8TZJ65FPH=GS2.1.s1773937369$o1$g0$t1773937369$j60$l0$h0; _ga=GA1.1.755914193.1752975774; _ga_5H1EHCT4DJ=GS2.1.s1773937345$o1$g1$t1773937373$j60$l0$h0; Culture=vi; DongTourHuongDan_1_dothang26905@gmail.com=1; vqc=1; ASP.NET_SessionId=k0ewlkd3wkjsgtooy10xk0qd; ruirophaply-covi19=30; Z_Hovering_Pu_hour_2={"count":0}; __gads=ID=46950cc340e16836:T=1773937347:RT=1780127310:S=ALNI_MahiPL2i1EMI2dkF0vEC4ztShphWA; __gpi=UID=00001224c46b5181:T=1773937347:RT=1780127310:S=ALNI_Mava0Hobg5uNP3wO_Mii9HkXV2HJQ; __eoi=ID=80bca3988c181074:T=1773937347:RT=1780127310:S=AA-AfjZwW-CjQDcKH8aReLnqZvVZ; __utmt=1; Cookie_VB=close; __utma=19472893.755914193.1752975774.1780128176.1780128176.1; __utmc=19472893; __utmz=19472893.1780128176.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); ticket=; thuvienphapluatnew=BDC8C3818448F5D7FBBFFD6D11FC6C944E469D5E8A3E79CDDB5A6B5517280645AED77697F8864E187A54F3485997AAB0A6890EFFEF3DF77A65C1FC489DB0E6362F9EDB42276C5645544DEEC47BDB10CA85D15CDB361C792D2CBADDE8E7BF3CFC9C7C290F440B6F3F33E5E0B4D8E93C8FD7358DA040C27B0B1B77EF2429F2410EF91EC95D9DA70A01E51A1523E54F19366982330A223487DFFB458AF9661B18A9C231E3C68F6B72BFBE2A77E27C5043AA24E39802816FEB76BDA296CC29FA231D06244C5344A69222DB287356CEB56CDDFC79345C647F5EE3EEABCDF0; dl_user=0=c5MGFHRnVaekkyT1RBMVFHZHRZV2xzTG1OdmJRPTWk; lg_user===c5MGFHRnVaekkyT1RBMVFHZHRZV2xzTG1OdmJTeFVRaXhHWVd4elpTdzJNemMyTURJeQWk; memberga=dothang26905@gmail.com[Free][6376022]; c_user=E1pNM05qQXlNaXd5TURJMkx6QTFMek13SURFMU9qRXpPakTm; _ga_PGVTRDMJGD=GS2.1.s1780123232$o18$g1$t1780128665$j34$l0$h0; _ga_BJ8R96BC8C=GS2.1.s1780123232$o18$g1$t1780128665$j34$l0$h0; _ga_E8YMVYW6Y9=GS2.1.s1780123232$o17$g1$t1780128665$j39$l0$h1243068048; __utmb=19472893.12.8.1780128665528; FCCDCF=%5Bnull%2Cnull%2Cnull%2Cnull%2Cnull%2Cnull%2C%5B%5B32%2C%22%5B%5C%22d1eeb7cb-5094-41f5-b69b-1ff9692e8c82%5C%22%2C%5B1773937346%2C209000000%5D%5D%22%5D%5D%5D; FCNEC=%5B%5B%22AKsRol_A_FQdfsrIlOK0mZWj6baHbw7-AGvb5vGvha22c5Ljk0Bj6C78Ajw-4LY_UhtyZ_IR-ZF7WuK4f5kVLOrPwIoChA2pK7oB6JlrvVS4LPJ1HE5ubnmLCZczYzL1Ow7HRILhUv5WhL0E9SbUpPjYJMXlPUUDJA%3D%3D%22%5D%5D'

cookies_dict = {}
for item in COOKIE_STRING.split(';'):
    if '=' in item:
        k, v = item.strip().split('=', 1)
        cookies_dict[k] = v
session.cookies.update(cookies_dict)

# 2. HEADER TỐI GIẢN, SẠCH SẼ (Gỡ bỏ toàn bộ Sec-Fetch-*)
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/148.0.0.0 Safari/537.36 Edg/148.0.0.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8',
    'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
})

BASE_URL = "https://thuvienphapluat.vn"

def read_keywords(filepath):
    if not os.path.exists(filepath):
        print(f"Không tìm thấy file {filepath}")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def get_document_links(keyword):
    encoded_keyword = quote(keyword)
    search_url = f"http://thuvienphapluat.vn/page/tim-van-ban.aspx?keyword={encoded_keyword}&area=0&type=40&status=0&lan=1&org=0&signer=0&match=True&sort=1&bdate=30/05/1946&edate=30/05/2026"
    
    print(f"\n----------------------------------------")
    print(f"Đang tìm kiếm chủ đề: {keyword}")
    try:
        response = session.get(search_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for a_tag in soup.find_all('a', href=True):
            raw_href = a_tag['href']
            clean_href = raw_href.split('?')[0] 
            
            if '/van-ban/' in clean_href and ('VBHN' in clean_href or 'Van-ban-hop-nhat' in clean_href):
                full_link = urljoin(BASE_URL, clean_href)
                print(f"-> Chọn văn bản đầu tiên: {full_link}")
                return [full_link] 
                
        print(f"-> Không tìm thấy văn bản hợp nhất nào cho '{keyword}'")
        return []
    except Exception as e:
        print(f"Lỗi khi tìm kiếm {keyword}: {e}")
        return []

def download_document(doc_url, keyword):
    try:
        # Tải HTML trang chi tiết văn bản
        response = session.get(doc_url, timeout=15)
        response.raise_for_status()
        
        # SĂN LINK TẢI ẨN BẰNG REGEX
        match = re.search(r"__urldl\s*=\s*['\"](/documents/download\.aspx.*?)['\"]", response.text)
        if not match:
            match = re.search(r"__urldl\s*=\s*['\"](https?://files\.thuvienphapluat\.vn.*?)['\"]", response.text)
            
        if not match:
            print(f"   [Lỗi] Không tìm thấy mã ẩn tải file. (Tài khoản của bạn có thể không đủ quyền tải).")
            return
            
        href = match.group(1)
        download_url = urljoin(BASE_URL, href)

        # Cập nhật nguồn click (Referer) cho hợp lệ
        session.headers.update({'Referer': doc_url})

        # GỌI TẢI FILE: Lúc này requests sẽ tự động quản lý redirect sang files.thuvienphapluat.vn
        # và truyền đúng các Cookie được cho phép.
        file_response = session.get(download_url, timeout=30)
        file_response.raise_for_status()

        # Kiểm tra xem có bị TVPL đá về HTML không
        content_type = file_response.headers.get('Content-Type', '')
        if 'text/html' in content_type:
            print(f"   [Lỗi] Máy chủ yêu cầu xác thực. Cookie có thể đã hết hạn, vui lòng F5 web lấy lại!")
            return
        
        # Xử lý tên file nhị phân tải về
        content_disp = file_response.headers.get('Content-Disposition')
        if content_disp and 'filename=' in content_disp:
            filename = content_disp.split('filename=')[-1].strip('"\'').encode('iso-8859-1').decode('utf-8', 'ignore')
        else:
            ext = ".doc" if "docx=" in download_url else ".pdf"
            doc_name = doc_url.split('/')[-1].replace('.aspx', '')
            filename = f"{doc_name}{ext}"

        filepath = os.path.join(RAW_DIR, filename)
        
        # Lưu file
        with open(filepath, 'wb') as f:
            f.write(file_response.content)
            
        print(f"   [Thành công] Đã tải về data/raw: {filename}")
        
        # Tạm nghỉ 3-6 giây chống Block
        time.sleep(random.uniform(3.5, 6.0)) 
        
    except requests.exceptions.HTTPError as err:
        if err.response.status_code == 403:
             print(f"   [Lỗi 403] Tường lửa chặn tải file. Vui lòng lấy lại Cookie mới.")
        else:
             print(f"   [Lỗi] HTTP Error: {err}")
    except Exception as e:
        print(f"   [Lỗi] Lỗi kết nối khi tải: {e}")

def main():
    keywords = read_keywords(CHUDE_FILE)
    if not keywords:
        print(f"Danh sách chủ đề trống. Vui lòng thêm từ khóa vào file {CHUDE_FILE}.")
        return

    for keyword in keywords:
        doc_links = get_document_links(keyword)
        if doc_links:
            download_document(doc_links[0], keyword)

if __name__ == "__main__":
    main()