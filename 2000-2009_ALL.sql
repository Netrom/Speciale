SELECT d.* 
FROM deals d 
WHERE 
		acq_nat="US" and 
		tar_nat="US" AND 
		status="C" AND 
		tar_pstat = 1 AND
		deal_no = 962266020 AND
		date_announced BETWEEN "2000-01-01" AND "2002-12-31" AND
		acq_gvkey IS NOT Null AND 
		tar_gvkey IS NOT Null AND 
		acq_type = "CORPORATE" AND 
		acq_naics is not null AND 
		tar_naics is not null AND 
		substring(LPAD(tar_sic, 4, "0"), 1, 2) = substring(LPAD(acq_sic, 4, "0"), 1, 2) AND 
		d.deal_no not in (
				SELECT deal_no FROM deals_tech WHERE tech_id IN (
				SELECT tech_id FROM deals_tech_def WHERE merger = 0)) AND 
		form in ("Merger", "Acquisition", "Acq. Maj. Int.", "Acq. of Assets") 
ORDER BY deal_no ASC