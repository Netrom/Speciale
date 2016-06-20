CREATE PROCEDURE `new_procedure` ()
BEGIN
SELECT d.*, h.hoda
FROM deals d 
LEFT join hoda h 
ON d.deal_no = h.deal_no 
WHERE 
		acq_nat="US" and 
		tar_nat="US" AND 
		status="C" AND 
		date_announced BETWEEN "2000-01-01" AND "2009-12-31" AND
		acq_gvkey > 0 AND
		tar_gvkey > 0 AND
		acq_type = "CORPORATE" AND
		#deal_val > 50 AND
		#acq_naics is not null AND
		#substring(LPAD(tar_sic, 4, "0"), 1, 1) != 6 AND
		#substring(LPAD(acq_sic, 4, "0"), 1, 1) != 6 AND
		#substring(LPAD(tar_sic, 4, "0"), 1, 2) = substring(LPAD(acq_sic, 4, "0"), 1, 2) AND
		d.deal_no not in (
				SELECT deal_no FROM deals_tech WHERE tech_id IN (
				SELECT tech_id FROM deals_tech_def WHERE merger = 0)) AND 
		form in ("Merger", "Acquisition", "Acq. Maj. Int.", "Acq. of Assets") 
ORDER BY date_announced ASC
END
