import struct
import sys
import json
import csv

def sample_dbf(dbf_path, n_records=5, output_csv=None):
    print(f"Reading DBF: {dbf_path}")
    with open(dbf_path, 'rb') as f:
        header = f.read(32)
        if len(header) < 32:
            print("Invalid DBF file (too short header).")
            return
        
        db_type, yy, mm, dd = struct.unpack('<BBBB', header[0:4])
        num_records = struct.unpack('<I', header[4:8])[0]
        header_len = struct.unpack('<H', header[8:10])[0]
        record_len = struct.unpack('<H', header[10:12])[0]
        
        print(f"DBF Type: {db_type:#x}")
        print(f"Last Update: 20{yy:02d}-{mm:02d}-{dd:02d}")
        print(f"Total Records: {num_records}")
        print(f"Header Length: {header_len}")
        print(f"Record Length: {record_len}")
        
        fields = []
        f.seek(32)
        while True:
            marker = f.read(1)
            if not marker or marker == b'\x0d':
                break
            field_desc = marker + f.read(31)
            if len(field_desc) < 32:
                break
            
            name = field_desc[0:11].split(b'\x00')[0].strip().decode('ascii', errors='ignore')
            field_type = chr(field_desc[11])
            field_len = field_desc[16]
            field_decimals = field_desc[17]
            
            fields.append({
                'name': name,
                'type': field_type,
                'len': field_len,
                'decimals': field_decimals
            })
            
        print(f"Found {len(fields)} fields:")
        for idx, field in enumerate(fields):
            if idx < 15:
                print(f"  {field['name']} ({field['type']}, length={field['len']})")
        if len(fields) > 15:
            print(f"  ... and {len(fields) - 15} more fields")
            
        f.seek(header_len)
        
        records_to_read = min(num_records, n_records)
        records = []
        for i in range(records_to_read):
            record_data = f.read(record_len)
            if len(record_data) < record_len:
                print(f"Warning: File ended prematurely at record {i}")
                break
            
            del_flag = chr(record_data[0])
            record = {}
            offset = 1
            for field in fields:
                field_bytes = record_data[offset : offset + field['len']]
                value_str = field_bytes.decode('latin1', errors='ignore').strip()
                record[field['name']] = value_str
                offset += field['len']
            
            records.append(record)
            
        print(f"\nSample of {len(records)} records:")
        for idx, r in enumerate(records):
            print(f"Record {idx + 1}:")
            keys_to_print = [f['name'] for f in fields[:15]]
            sample_data = {k: r[k] for k in keys_to_print if k in r}
            print(json.dumps(sample_data, indent=2))
            if len(fields) > 15:
                print(f"  ... and {len(fields) - 15} more fields")
                
        if output_csv and records:
            with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=[f['name'] for f in fields])
                writer.writeheader()
                for r in records:
                    writer.writerow(r)
            print(f"\nSaved sample to {output_csv}")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 sample_dbf.py <dbf_path> [n_records] [output_csv]")
    else:
        path = sys.argv[1]
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        out = sys.argv[3] if len(sys.argv) > 3 else None
        sample_dbf(path, n, out)
