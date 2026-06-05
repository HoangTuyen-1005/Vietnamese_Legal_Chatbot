from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
import os

default_args = {
    'owner': 'do_nhat_thang',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'vietnam_legal_rag_etl_pipeline',
    default_args=default_args,
    description='Hệ thống trọn gói ETL: Cào dữ liệu -> Ingest (Cleaner & Chunker) -> Build Index Vector',
    schedule_interval='0 0 * * *',  # Tự động chạy lúc 00:00 hàng ngày
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['legal_chatbot', 'full_pipeline'],
) as dag:

    # 🟢 TASK 1: Cào dữ liệu pháp luật (Giữ nguyên cấu trúc đã chạy thành công)
    task_1_crawl_data = BashOperator(
        task_id='1_Crawl_Law_Documents',
        bash_command='cd /opt/airflow && python3 data_pipeline/dags/crawler.py',
        env={
            'RAW_DATA_DIR': '/opt/airflow/data_pipeline/data/raw',
            **os.environ
        }
    )

    # 🔵 TASK 2: Ingest dữ liệu (Ép PYTHONPATH trực tiếp vào lệnh bash)
    task_2_ingest_data = BashOperator(
        task_id='2_Ingest_Documents',
        bash_command='export PYTHONPATH=/opt/airflow:/opt/airflow/data_pipeline && cd /opt/airflow && python3 data_pipeline/scripts/ingest_pdf.py',
    )

   # 🟡 TASK 3: Build Index
    task_3_build_index = BashOperator(
        task_id='3_Build_Vector_Index',
        bash_command='export PYTHONPATH=/opt/airflow:/opt/airflow/data_pipeline && cd /opt/airflow && python3 data_pipeline/scripts/build_index.py',
        env={
            # Dùng 'host.docker.internal' để Docker có thể nhìn xuyên ra máy thật của bạn
            'QDRANT_HOST': 'host.docker.internal',
            **os.environ
        }
    )

    # ĐỊNH NGHĨA LUỒNG CHẢY TUẦN TỰ (Cào xong -> Trích xuất/Băm nhỏ -> Đẩy lên Kho lưu trữ)
    task_1_crawl_data >> task_2_ingest_data >> task_3_build_index
