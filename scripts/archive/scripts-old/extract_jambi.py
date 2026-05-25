import struct
import sys
import os
import csv

def extract_province(dbf_path, output_csv, prov_code="15"):
    print(f"Scanning {dbf_path} for Province {prov_code}...")
    if not os.path.exists(dbf_path):
        print(f"Error: {dbf_path} does not exist.")
        return

    with open(dbf_path, 'rb') as f:
        header = f.read(32)
        if len(header) < 32:
            print("Invalid DBF header.")
            return
            
        num_records = struct.unpack('<I', header[4:8])[0]
        header_len = struct.unpack('<H', header[8:10])[0]
        record_len = struct.unpack('<H', header[10:12])[0]
        
        fields = []
        f.seek(32)
        while True:
            marker = f.read(1)
            if not marker or marker == b'\x0d':
                break
            field_desc = marker + f.read(31)
            name = field_desc[0:11].split(b'\x00')[0].strip().decode('ascii', errors='ignore')
            field_type = chr(field_desc[11])
            field_len = field_desc[16]
            fields.append({'name': name, 'type': field_type, 'len': field_len})
            
        r101_offset = 1
        r101_len = 0
        r101_found = False
        
        current_offset = 1
        for field in fields:
            if field['name'] == 'R101':
                r101_offset = current_offset
                r101_len = field['len']
                r101_found = True
            current_offset += field['len']
            
        if not r101_found:
            print("Error: R101 field not found in DBF.")
            return
            
        print(f"R101 offset: {r101_offset}, length: {r101_len}")
        f.seek(header_len)
        
        matched_records = []
        chunk_size = 10000
        total_processed = 0
        while total_processed < num_records:
            records_left = num_records - total_processed
            current_chunk = min(chunk_size, records_left)
            chunk_data = f.read(record_len * current_chunk)
            if not chunk_data:
                break
                
            for i in range(current_chunk):
                rec_start = i * record_len
                r101_bytes = chunk_data[rec_start + r101_offset : rec_start + r101_offset + r101_len]
                r101_val = r101_bytes.decode('latin1', errors='ignore').strip()
                
                if r101_val == prov_code:
                    record = {}
                    offset = 1
                    for field in fields:
                        field_bytes = chunk_data[rec_start + offset : rec_start + offset + field['len']]
                        record[field['name']] = field_bytes.decode('latin1', errors='ignore').strip()
                        offset += field['len']
                    matched_records.append(record)
                    
            total_processed += current_chunk
            if total_processed % 100000 == 0 or total_processed == num_records:
                print(f"  Processed {total_processed}/{num_records} records... Matched so far: {len(matched_records)}")
                
        if matched_records:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[f['name'] for f in fields])
                writer.writeheader()
                writer.writerows(matched_records)
            print(f"Successfully extracted {len(matched_records)} records to {output_csv}\n")
        else:
            print(f"No records found for Province {prov_code}")

if __name__ == '__main__':
    # Determine the script directory to find data folder relatively
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data")
    
    rt_dbf = os.path.join(data_dir, "ssn202403_kor_rt.dbf")
    ind_dbf = os.path.join(data_dir, "ssn202403_kor_ind1.dbf")
    
    rt_csv = os.path.join(data_dir, "ssn202403_kor_rt_jambi.csv")
    ind_csv = os.path.join(data_dir, "ssn202403_kor_ind1_jambi.csv")
    
    extract_province(rt_dbf, rt_csv, "15")
    extract_province(ind_dbf, ind_csv, "15")
