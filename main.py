import os
import json
import requests
import time
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
import logging

def setup_logging(output_dir):
    """Thiết lập logging"""
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Log file cho mỗi lần chạy
    run_log = os.path.join(log_dir, f'conversion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(run_log, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

def load_processed_files(output_dir):
    """Đọc danh sách các file đã xử lý"""
    processed_files_log = os.path.join(output_dir, 'processed_files.json')
    if os.path.exists(processed_files_log):
        with open(processed_files_log, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'processed_files': [], 'last_update': None}

def save_processed_file(output_dir, filename):
    """Lưu tên file đã xử lý vào log"""
    processed_files_log = os.path.join(output_dir, 'processed_files.json')
    processed_data = load_processed_files(output_dir)
    
    if filename not in processed_data['processed_files']:
        processed_data['processed_files'].append(filename)
        processed_data['last_update'] = datetime.now().isoformat()
        
        with open(processed_files_log, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

def generate_with_ollama(prompt, model="llama3.2"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            return response.json()["response"]
        return None
    except Exception as e:
        logging.error(f"Error calling Ollama: {e}")
        return None

def chunk_text(text, chunk_size=1000):
    """Chia văn bản thành các đoạn nhỏ hơn"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_chunk.append(word)
        current_size += len(word) + 1
        if current_size >= chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
            
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

def convert_chunk_to_format(chunk):
    """Chuyển đổi một đoạn text thành format mong muốn"""
    prompt = f"""
    Dưới đây là một đoạn văn bản. Chuyển đổi nó thành 2-3 mẫu dùng để training.
    Mỗi mẫu phải ở định dạng này:
    {{
        "instruction": "Một câu hỏi hoặc các yêu cầu về nội dung",
        "input": "Ngữ cảnh liên quan từ văn bản (điều này là tùy chọn)",
        "output": "Câu trả lời 1 cách chi tiết hoặc hoàn chỉnh"
    }}
    
    Text: {chunk}
    
    Chỉ trả về mảng JSON chứa các mẫu, không có gì khác.
    """
    
    response = generate_with_ollama(prompt)
    try:
        samples = json.loads(response)
        if isinstance(samples, list):
            return samples
        return None
    except:
        return None

def process_single_file(input_file_path, output_dir):
    """Xử lý một file và lưu kết quả vào file riêng"""
    try:
        input_filename = Path(input_file_path).stem
        output_file_path = os.path.join(output_dir, f"{input_filename}_processed.jsonl")
        temp_file_path = os.path.join(output_dir, f"{input_filename}_temp.jsonl")
        
        logging.info(f"Starting to process {input_filename}")
        
        # Đọc file input
        with open(input_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Chia thành chunks
        chunks = chunk_text(text)
        processed_samples = 0
        
        # Sử dụng file tạm thời
        with open(temp_file_path, 'w', encoding='utf-8') as temp_file:
            for chunk in tqdm(chunks, desc=f"Processing chunks of {input_filename}"):
                try:
                    samples = convert_chunk_to_format(chunk)
                    if samples:
                        for sample in samples:
                            temp_file.write(json.dumps(sample, ensure_ascii=False) + '\n')
                            temp_file.flush()
                            processed_samples += 1
                except Exception as e:
                    logging.error(f"Error processing chunk in {input_filename}: {str(e)}")
                    continue
                
                time.sleep(1)
        
        # Đổi tên file tạm thành file chính
        if os.path.exists(output_file_path):
            os.remove(output_file_path)
        os.rename(temp_file_path, output_file_path)
        
        # Lưu vào log file đã xử lý
        save_processed_file(output_dir, input_filename)
        
        logging.info(f"Completed {input_filename}: {processed_samples} samples generated")
        return processed_samples
        
    except Exception as e:
        logging.error(f"Error processing file {input_filename}: {str(e)}")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        return 0

def process_directory(input_dir, output_dir):
    """Xử lý tất cả file trong thư mục input"""
    # Tạo thư mục output và thiết lập logging
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    
    # Đọc danh sách file đã xử lý
    processed_data = load_processed_files(output_dir)
    processed_files = processed_data['processed_files']
    
    if processed_files:
        logging.info(f"Found {len(processed_files)} previously processed files")
    
    total_files = 0
    total_samples = 0
    skipped_files = 0
    
    # Xử lý từng file
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_filename = Path(filename).stem
            
            # Skip nếu file đã được xử lý
            if input_filename in processed_files:
                logging.info(f"Skipping {filename} (already processed)")
                skipped_files += 1
                continue
                
            input_file_path = os.path.join(input_dir, filename)
            samples_count = process_single_file(input_file_path, output_dir)
            
            total_files += 1
            total_samples += samples_count
    
    logging.info("\nProcessing completed:")
    logging.info(f"Total files processed: {total_files}")
    logging.info(f"Files skipped: {skipped_files}")
    logging.info(f"Total samples generated: {total_samples}")

# Sử dụng
if __name__ == "__main__":
    input_directory = "/Users/justoanff/Desktop/data_converter/raw-texts"
    output_directory = "processed_data"
    
    process_directory(input_directory, output_directory)
