import csv
import os
from collections import Counter
import statistics

def compute_y(r1207, r1208):
    if r1207 in ('5', '2'):
        return 0
    elif r1207 == '1':
        try:
            batang = int(r1208)
            return 1 if batang >= 140 else 0
        except ValueError:
            return None
    return None

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    data_dir = os.path.join(project_dir, "data")
    
    rt_csv = os.path.join(data_dir, "ssn202403_kor_rt_jambi.csv")
    ind_csv = os.path.join(data_dir, "ssn202403_kor_ind1_jambi.csv")
    out_csv = os.path.join(data_dir, "processed_krt_jambi.csv")
    
    print("Membaca data Rumah Tangga (RT)...")
    rt_data = {}
    with open(rt_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Key gabungan
            key = (row.get('R101'), row.get('R102'), row.get('R105'), 
                   row.get('WI1'), row.get('WI2'), row.get('PSU'), 
                   row.get('SSU'), row.get('URUT'))
            rt_data[key] = row
            
    print("Membaca dan memproses data Individu (IND)...")
    merged_data = []
    
    # Fitur yang akan diekstrak
    # IND: Umur(R407), JK(R405), Kawin(R404), Pendidikan(R612), Kerja(R706), KeluhanSehat(R1102), Internet(R812)
    # RT: JmlART(R1801), UsahaMikro(R2210AA), Bansos(R2207), Wilayah(R105),
    #     StatusBangunan(R1802), LuasLantai(R1804), Lantai(R1808), Dinding(R1807), Atap(R1806A),
    #     Air(R1810A), Penerangan(R1816), BahanBakar(R1817), Motor(R2001H), Laptop(R2001F)
    
    features_num = ['umur_krt', 'jumlah_art', 'luas_lantai']
    features_cat = [
        'jk_krt', 'status_kawin_krt', 'pendidikan_krt', 'pekerjaan_krt', 'keluhan_kesehatan', 'internet_krt',
        'usaha_mikro', 'bansos', 'wilayah', 'status_bangunan', 'jenis_lantai', 'jenis_dinding', 'jenis_atap',
        'sumber_air', 'penerangan', 'bahan_bakar', 'motor', 'laptop', 'kabupaten'
    ]
    
    with open(ind_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('R403') == '1': # Hanya KRT
                y = compute_y(row.get('R1207', ''), row.get('R1208', ''))
                if y is None:
                    continue # Drop invalid target
                    
                key = (row.get('R101'), row.get('R102'), row.get('R105'), 
                       row.get('WI1'), row.get('WI2'), row.get('PSU'), 
                       row.get('SSU'), row.get('URUT'))
                       
                rt_row = rt_data.get(key, {})
                
                # Ekstrak fitur
                record = {
                    'Y': y,
                    'umur_krt': row.get('R407', ''),
                    'jk_krt': row.get('R405', ''),
                    'status_kawin_krt': row.get('R404', ''),
                    'pendidikan_krt': row.get('R612', ''),
                    'pekerjaan_krt': row.get('R706', ''),
                    'keluhan_kesehatan': row.get('R1102', ''),
                    'internet_krt': row.get('R812', ''),
                    
                    'jumlah_art': rt_row.get('R1801', ''),
                    'usaha_mikro': rt_row.get('R2210AA', ''),
                    'bansos': rt_row.get('R2207', ''),
                    'wilayah': rt_row.get('R105', ''),
                    'kabupaten': rt_row.get('R102', ''),
                    
                    'status_bangunan': rt_row.get('R1802', ''),
                    'luas_lantai': rt_row.get('R1804', ''),
                    'jenis_lantai': rt_row.get('R1808', ''),
                    'jenis_dinding': rt_row.get('R1807', ''),
                    'jenis_atap': rt_row.get('R1806A', ''),
                    'sumber_air': rt_row.get('R1810A', ''),
                    'penerangan': rt_row.get('R1816', ''),
                    'bahan_bakar': rt_row.get('R1817', ''),
                    'motor': rt_row.get('R2001H', ''),
                    'laptop': rt_row.get('R2001F', '')
                }
                merged_data.append(record)
                
    print(f"Total data KRT tergabung (dengan target valid): {len(merged_data)}")
    
    # Hitung nilai median dan modus untuk imputasi
    print("Menghitung nilai imputasi (Median untuk numerik, Modus untuk kategorik)...")
    impute_vals = {}
    
    for f_num in features_num:
        vals = []
        for r in merged_data:
            val = r[f_num]
            if val and val != '.':
                try:
                    vals.append(float(val))
                except ValueError:
                    pass
        impute_vals[f_num] = statistics.median(vals) if vals else 0
        
    for f_cat in features_cat:
        counter = Counter()
        for r in merged_data:
            val = r[f_cat]
            if val and val != '.':
                counter[val] += 1
        impute_vals[f_cat] = counter.most_common(1)[0][0] if counter else 'Unknown'
        
    print("Melakukan imputasi missing values...")
    # Imputasi
    for r in merged_data:
        for f_num in features_num:
            if r[f_num] == '' or r[f_num] == '.':
                r[f_num] = impute_vals[f_num]
        for f_cat in features_cat:
            if r[f_cat] == '' or r[f_cat] == '.':
                r[f_cat] = impute_vals[f_cat]
                
    print(f"Menyimpan data bersih ke {out_csv}...")
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['Y'] + features_num + features_cat
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(merged_data)
        
    print("Selesai! Data preprocessing berhasil.")

if __name__ == "__main__":
    main()
