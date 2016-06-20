	SELECT deal_no, date_announced, acq_name, acq_sic, acq_sic_crsp, acq_naics, acq_naics_crsp, stom.desc as tom, scrsp.desc as crsp, ss.to_naics as tom_conv, ss2.to_naics as crsp_conv FROM
		(SELECT n.*, mva.mv as mva, mvb.mv as mvb, mvb.mv/mva.mv as mv_rat FROM (
			SELECT *, (SELECT date FROM ff WHERE date < d.date_announced ORDER BY date DESC LIMIT 1 OFFSET 40) as t_1
			FROM
				(SELECT d.*, h.hoda
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
						acq_naics is not null AND
						deal_val > 50 AND
						substring(LPAD(tar_sic, 4, "0"), 1, 1) != 6 AND
						substring(LPAD(acq_sic, 4, "0"), 1, 1) != 6 AND
						substring(LPAD(tar_sic, 4, "0"), 1, 3) = substring(LPAD(acq_sic, 4, "0"), 1, 3) AND
						d.deal_no not in (
								SELECT deal_no FROM deals_tech WHERE tech_id IN (
								SELECT tech_id FROM deals_tech_def WHERE merger = 0)) AND 
						form in ("Merger", "Acquisition", "Acq. Maj. Int.", "Acq. of Assets")) d) n
			LEFT JOIN mv_cache mva ON
			n.acq_gvkey = mva.gvkey AND n.t_1 = mva.datadate
			LEFT JOIN mv_cache mvb ON
			n.tar_gvkey = mvb.gvkey AND n.t_1 = mvb.datadate) q
	LEFT JOIN sic_def stom ON q.acq_sic = stom.sic
    LEFT JOIN sic_def scrsp ON q.acq_sic_crsp = scrsp.sic
    LEFT JOIN f_cen_ct_sic ss ON ss.from_sic = q.acq_sic AND ss.to_year = 2002 AND ss.to_naics IN (SELECT to_naics FROM speciale.f_cen_ct_naics WHERE from_year = '2007' AND from_naics = q.acq_naics)
    LEFT JOIN f_cen_ct_sic ss2 ON ss2.from_sic = q.acq_sic_crsp AND ss2.to_year = 2002 AND (ss2.to_naics IN (SELECT to_naics from f_cen_ct_naics WHERE q.acq_naics_crsp = from_naics) or ss2.to_naics = q.acq_naics_crsp)
	WHERE mv_rat>0.2 and deal_no != 1023743020 
		#and substring(LPAD(acq_sic, 4, "0"), 1, 4) = substring(LPAD(acq_sic_crsp, 4, "0"), 1, 4) AND
		#acq_naics != acq_naics_crsp
	ORDER BY date_announced