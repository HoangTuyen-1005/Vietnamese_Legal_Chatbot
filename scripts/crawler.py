import os
import re
import time
import requests
import urllib.parse
from pathlib import Path
from bs4 import BeautifulSoup

# --- CẤU HÌNH ---
RAW_DIR = Path("data_pipeline/data/raw")
INPUT_FILE = "data_pipeline/ChuDe.txt" # File chứa danh sách chủ đề/tên luật cần cào

LISTING_URLS = [
    "https://vbpl.vn/TW/Pages/vanban.aspx?idLoaiVanBan=16&dvid=13", # Bộ luật
    # "https://vbpl.vn/TW/Pages/vanban.aspx?idLoaiVanBan=17&dvid=13"  # Luật
]

VBPL_TOANVAN_URL = "https://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID={}"
BASE_DOMAIN = "https://vbpl.vn"

# Tăng thời gian chờ lên 60 giây và cấu hình thử lại (Retry)
REQUEST_TIMEOUT = 60 
MAX_RETRIES = 3

def safe_get(session, url, stream=False):
    """Bọc request: Tự động thử lại nếu server bị timeout hoặc nghẽn mạng."""
    for attempt in range(MAX_RETRIES):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT, stream=stream)
            response.raise_for_status()
            return response
        except requests.exceptions.Timeout:
            print(f"    [!] Quá thời gian chờ. Thử lại lần {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(3)
        except requests.exceptions.RequestException as e:
            print(f"    [!] Lỗi mạng ({e}). Thử lại lần {attempt + 1}/{MAX_RETRIES}...")
            time.sleep(3)
    raise Exception(f"Không thể kết nối đến server sau {MAX_RETRIES} lần thử.")


def get_items_from_listing(session, base_url):
    """Quét tất cả các trang, lấy TOÀN BỘ text của block để filter"""
    all_items_in_category = {}
    page = 1
    
    while True:
        url = f"{base_url}&Page={page}"
        print(f"Đang quét Trang {page} tại: {url}")
        
        try:
            response = safe_get(session, url)
            soup = BeautifulSoup(response.text, 'html.parser')
            items_found_on_page = 0
            
            for title_p in soup.find_all('p', class_='title'):
                a_tag = title_p.find('a', href=True)
                if a_tag:
                    href = a_tag['href']
                    if "ItemID=" in href:
                        match = re.search(r"ItemID=(\d+)", href)
                        if match:
                            item_id = match.group(1)
                            short_title = a_tag.get_text(strip=True)
                            
                            container = title_p.find_parent('li')
                            if not container:
                                container = title_p.parent
                            
                            # Lấy toàn bộ text để dùng cho bộ lọc
                            full_text = container.get_text(separator=" ", strip=True)
                            
                            # Tạo tên file đẹp (Ghép dòng 1 và dòng 2)
                            des_div = container.find('div', class_='des') if container else None
                            des_text = des_div.get_text(strip=True) if des_div else ""
                            display_title = f"{short_title} - {des_text}" if des_text else short_title
                            
                            # LƯU Ý: Phải lưu dưới dạng dictionary để tránh lỗi "string indices must be integers"
                            if item_id not in all_items_in_category: 
                                all_items_in_category[item_id] = {
                                    'display_title': display_title,
                                    'full_text': full_text
                                }
                                items_found_on_page += 1
                                
            if items_found_on_page == 0:
                print("  -> Đã quét đến trang cuối cùng.")
                break
                
            page += 1
            time.sleep(1.5)
            
        except Exception as e:
            print(f"  [X] Dừng lật trang do lỗi: {e}")
            break
            
    return all_items_in_category

def get_real_download_links(session, item_id):
    """Dò UI, bóc tách link tải thật (xử lý hàm JavaScript ẩn)"""
    url = VBPL_TOANVAN_URL.format(item_id)
    links = []
    
    def extract_actual_link(href):
        if "javascript:downloadfile" in href:
            match = re.search(r"downloadfile\s*\(\s*'[^']+'\s*,\s*'([^']+)'\s*\)", href)
            if match:
                path = match.group(1)
                safe_path = urllib.parse.quote(path, safe='/')
                return BASE_DOMAIN + safe_path
            return None
        if any(ext in href.lower() for ext in ['.doc', '.docx', '.pdf']):
            return href if href.startswith('http') else BASE_DOMAIN + href
        return None

    try:
        response = safe_get(session, url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        xem_nhanh_elements = soup.find_all(string=re.compile(r"\(Xem nhanh\)", re.IGNORECASE))
        if xem_nhanh_elements:
            for element in xem_nhanh_elements:
                container = element.find_parent(['li', 'div', 'p', 'tr'])
                if container:
                    for a_tag in container.find_all('a', href=True):
                        href = a_tag['href']
                        text = a_tag.get_text(strip=True)
                        if "xem nhanh" in text.lower() or "xemnhanh" in href.lower():
                            continue
                        
                        full_link = extract_actual_link(href)
                        if full_link:
                            if "Web=0" not in full_link:
                                full_link += "&Web=0" if "?" in full_link else "?Web=0"
                            if full_link not in links:
                                links.append(full_link)
        
        if not links:
            for a_tag in soup.find_all('a', href=True):
                full_link = extract_actual_link(a_tag['href'])
                if full_link and "/Attachments/" in full_link:
                    if "Web=0" not in full_link:
                        full_link += "&Web=0" if "?" in full_link else "?Web=0"
                    if full_link not in links:
                        links.append(full_link)
                        
        return links
    except Exception as e:
        return []

def download_binary_file(session, url, item_id, title):
    """Tải file nhị phân và kiểm tra an toàn HTML SharePoint"""
    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1]
    if not ext:
        ext = ".doc"
        
    safe_title = re.sub(r'[\\/*?:"<>|]', "", title).strip()
    if len(safe_title) > 100:
        safe_title = safe_title[:100] + "..."
        
    filename = f"{item_id} - {safe_title}{ext}"
    output_path = RAW_DIR / filename

    if output_path.exists():
        return "skip", filename

    response = safe_get(session, url, stream=True)

    content_type = response.headers.get('Content-Type', '').lower()
    if 'text/html' in content_type:
        raise Exception("Server trả về HTML (Lỗi WopiFrame) thay vì file gốc.")

    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            
    return "ok", filename

def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    if not os.path.exists(INPUT_FILE):
        print(f"Lỗi: Không tìm thấy file '{INPUT_FILE}'.")
        return
        
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        topics = [line.strip().lower() for line in f.readlines() if line.strip()]

    print(f"Đã nạp {len(topics)} chủ đề từ '{INPUT_FILE}': {', '.join(topics)}")
    print("-" * 50)

    with requests.Session() as session:
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0.0.0 Safari/537.36"
        })
        
        all_items = {}
        for url in LISTING_URLS:
            all_items.update(get_items_from_listing(session, url))
            time.sleep(1)
            
        print(f"\n=> Đã thu thập dữ liệu {len(all_items)} văn bản từ các trang danh mục.")

        # 3. Màng lọc chủ đề thông minh (Word-based Match)
        filtered_items = {}
        for item_id, data in all_items.items():
            full_text_lower = data['full_text'].lower()
            
            for topic in topics:
                topic_words = topic.split()
                if all(word in full_text_lower for word in topic_words):
                    filtered_items[item_id] = data['display_title']
                    break 
                
        print(f"=> Lọc thành công: Có {len(filtered_items)} văn bản khớp với chủ đề yêu cầu.\n" + "-"*50)

        for idx, (item_id, title) in enumerate(filtered_items.items(), start=1):
            print(f"[{idx}/{len(filtered_items)}] Đang xử lý: {title} (ID: {item_id})")
            
            file_links = get_real_download_links(session, item_id)
            
            if not file_links:
                print(f"  -> Bỏ qua: Không tìm thấy file đính kèm.")
                continue

            try:
                first_link = file_links[0]
                status, fname = download_binary_file(session, first_link, item_id, title)
                if status == "ok":
                    print(f"  -> Đã tải: {fname}")
                elif status == "skip":
                    print(f"  -> Đã tồn tại: {fname}")
            except Exception as e:
                print(f"  -> Lỗi khi tải: {e}")
            
            time.sleep(1.5)

    print("-" * 50)
    print("HOÀN TẤT CÀO DỮ LIỆU CHUẨN!")

if __name__ == "__main__":
    main()