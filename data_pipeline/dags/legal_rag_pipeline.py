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
    description='Hệ thống Hybrid ETL: Cào dữ liệu tại Local bằng IP nhà dân',
    schedule_interval='0 0 * * *',  # Chạy tự động lúc 00:00 hàng ngày
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['legal_chatbot', 'crawler', 'local'],
) as dag:

    # TASK DUY NHẤT: Gọi script cào dữ liệu chạy an toàn trong Docker Local
    task_1_crawl_data = BashOperator(
        task_id='Crawl_Law_Documents_Local',
        # Ép đứng đúng thư mục gốc chứa toàn bộ dự án để đọc được ChuDe.txt
        bash_command='cd /opt/airflow && python3 data_pipeline/dags/crawler.py',
        env={
            # Trỏ đúng vào ngách thư mục data/raw đã map từ máy bạn vào Docker
            'RAW_DATA_DIR': '/opt/airflow/data_pipeline/data/raw',
            'TVPL_USERNAME': 'dothang26905@gmail.com',
            'TVPL_PASSWORD': 'dnt2692005',
            **os.environ
        }
    )

    task_1_crawl_data