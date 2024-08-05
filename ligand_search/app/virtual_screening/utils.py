from django.conf import settings
from datetime import datetime
import os

def is_valid_pdbqt(file):
    # pdbqt 파일이 올바른지 확인하는 함수
    file.seek(0)  # 파일 포인터를 처음으로 돌림
    try:
        for line in file:
            if line.startswith(b'REMARK') or line.startswith(b'ATOM') or line.startswith(b'HETATM'):
                return True
        return False
    except Exception as e:
        return False

def get_ligand_name(file):
    # 파일에서 리간드 이름을 추출하는 함수
    file.seek(0)  # 파일 포인터를 처음으로 돌림
    try:
        for line in file:
            line_str = line.decode('utf-8')
            if "Name" in line_str:
                # "Name =" 뒤의 값을 추출
                name_part = line_str.split('Name')[1].split('=')[1].strip()
                
                # 이름 공백 예외 처리
                if not name_part:
                    return None
                
                return name_part
        return None
    except Exception as e:
        return None

def get_unique_file_name(file_name):
    # 파일 이름 중복 방지
    # 기본 파일 이름 + timestamp (+ 혹시나 이렇게 해도 동일 파일 존재시 예외 처리용 숫자) + .pdbqt
    base_name, ext = os.path.splitext(file_name)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    unique_file_name = f"{base_name}_{timestamp}{ext}"
    counter = 1

    while os.path.exists(os.path.join(settings.LIGAND_FILE_PATH, unique_file_name)):
        unique_file_name = f"{base_name}_{timestamp}_{counter}{ext}"
        counter += 1

    return unique_file_name