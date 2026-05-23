# AI Context: Susenas KOR (March 2024) Metadata Schema

> **Dataset Overview:** This metadata defines the schema for the March 2024 Indonesian National Socio-Economic Survey (Susenas KOR). It spans three structural levels of data collection: Household (`RT`), Individual (`IND`), and Migration/Ex-Household Members (`MIG`).

## 1. Geographic & Base Identifiers (Common Across Levels)

These variables establish sampling locations and link datasets together.

* `URUT`: Household renumbering identifier (`nurt`).
* `WI1` / `WI2`: Sampling renumbering keys (`nks` / `nurt`).
* `PSU` / `SSU` / `Strata`: Sampling units and stratification keys.
* `R101` (**Province**):
* `11`: Aceh | `12`: Sumatera Utara | `13`: Sumatera Barat | `14`: Riau | `15`: Jambi | `16`: Sumatera Selatan | `17`: Bengkulu | `18`: Lampung | `19`: Kepulauan Bangka Belitung | `21`: Kepulauan Riau | `31`: DKI Jakarta | `32`: Jawa Barat | `33`: Jawa Tengah | `34`: DI Yogyakarta | `35`: Jawa Timur | `36`: Banten | `51`: Bali | `52`: Nusa Tenggara Barat | `53`: Nusa Tenggara Timur | `61`: Kalimantan Barat | `62`: Kalimantan Tengah | `63`: Kalimantan Selatan | `64`: Kalimantan Timur | `65`: Kalimantan Utara | `71`: Sulawesi Utara | `72`: Sulawesi Tengah | `73`: Sulawesi Selatan | `74`: Sulawesi Tenggara | `75`: Gorontalo | `76`: Sulawesi Barat | `81`: Maluku | `82`: Maluku Utara | `91`: Papua Barat | `92`: Papua Barat Daya | `94`: Papua | `95`: Papua Selatan | `96`: Papua Tengah | `97`: Papua Pegunungan


* `R102`: Regency/City (**Kabupaten/Kota**) numeric codes.
* `R105` (**Region Type**):
* `1`: Perkotaan (Urban) | `2`: Perdesaan (Rural)

---

## 2. Household Level Schema (`RT`)

Focuses on food security, housing characteristics, social assistance program absorption, and micro-business ownership.

### Food Security (Block 17)

* `NUINFORT`: Household informant line number.
* `R1701` to `R1708` use the standard answer key: `1` = Ya (Yes), `5` = Tidak (No), `8` = Tidak tahu (Don't Know), `9` = Menolak (Refused).
* `R1701`: Worried about not having enough food.
* `R1702`: Unable to eat healthy/nutritious food.
* `R1703`: Ate only a few kinds of foods.
* `R1704`: Skipped a meal.
* `R1705`: Ate less than they felt they should.
* `R1706`: Ran out of food.
* `R1707`: Hungry but did not eat.
* `R1708`: Went without eating for a whole day.



### Housing & Assets (Block 18)

* `R1801`: Number of families living in the building.
* `R1802` (**Housing Ownership Status**): *e.g., Own house, rent, official housing.*
* `R1803`: Type of land/property ownership proof.

### Social Assistance & Programs (Block 22)

* `R2208E5_A` to `R2208E5_G`: Intended use of BPNT (Non-Cash Food Aid) funds received in October 2023.
* `_A`: School fees | `_F`: Loan/credit repayment | `_G` / `LAI`: Others.


* `R2209A`: Received Village BLT (Direct Cash Assistance).
* `R2209B`: Received PKTD (Cash for Work).
* `R2209C`: Received food/rice assistance.
* `R2209D`: Received land certification assistance.
* `R2211A`: Received regular local government cash transfers.
* `R2211A1`: Food transfer value | `R2211A2`: Cash transfer value.



### Micro/Small Business Ownership

* `R2210AA`: Does any household member own a micro or small enterprise? (`1` = Yes, `5` = No)
* `R2210B1` to `R2210B5`: Technical/business support utilization tracker.
* `B1`: Production training | `B2`: Commercial licensing | `B3`: Digital marketing | `B4`: Financial reporting | `B5`: Financing facilities.



---

## 3. Individual Level Schema (`IND`)

Demographics, education, labor, health, fertility, and individual-targeted social aid.

### Demographics

* `R401`: Individual line number within the household (`Nomor urut art`).
* `R403` (**Relationship to Household Head**):
* `1`: Kepala rumah tangga (Head) | `2`: Istri/suami (Spouse) | `3`: Anak kandung/tiri (Child) | `4`: Anak angkat (Adopted child) | `5`: Menantu (Child-in-law) | `6`: ... [Truncated]


* `R404`: Marital status.
* `R405` (**Gender**): `1` = Laki-laki (Male), `2` = Perempuan (Female).
* `R406A / B / C`: Date / Month / Year of birth.
* `R407`: Age (`Umur`).
* `R503` / `R504`: Line numbers of biological mother / father within the household (for family mapping).

### Fertility & Health (Block 15 & 16)

* `R1501`: Has the female member ever given birth to a live-born child?
* `R1502B`: Age at first birth.
* `R1503`: Date of last live birth.
* `R1504A` / `LAIN`: Location of the last birth (and write-in options).
* `R1504B`: Birth attendant/assistant identity.
* `R1504C`: Infant birth weight.
* `R1505`: Duration of immediate skin-to-skin contact (initiation of breastfeeding).
* `R1601`: Ever/currently using family planning methods (KB / traditional).
* `R1602`: Current contraceptive/family planning method used.
* `R1603`: Supply source for modern family planning methods.

### Targeted Social Assistance

* `R2205A`: Individual receives aid from the Elderly Attention Program (`bansos atensi lansia`).
* `R2205B`: Line number corresponding to the elderly aid recipient.
* `R2210AB`: Line number corresponding to the micro/small business owner.

---

## 4. Migration & International Workers Level Schema (`MIG`)

Tracks former household members currently residing abroad.

* `R2301`: Are there any former household members living overseas? (`1` = Yes, `5` = No)
* **Individual records loop via suffixes (`B1`, `B2`, `B3`) per household:**
* `R2302B1`: Line number of the former household member.
* `R2304B1` (**Gender**): `1` = Male, `2` = Female.
* `R2305B1`: Destination country of residence.
* `R2306B1`: Year of departure.
* `R2307B1`: Age at departure.
* `R2308B1` (**Education level at departure**): `1` = Paket A/Elementary equivalents...
* `R2309B1`: Primary reason for migration.
