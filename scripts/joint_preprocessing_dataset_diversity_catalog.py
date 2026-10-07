"""Freeze a conservative source-group inventory from public catalog snapshots.

This script uses metadata only. It never loads model predictions or chooses data
using observed preprocessing gains. Source groups are provenance safeguards,
not learned similarity clusters. Some older OpenML classification versions use
discretized targets on real observations; this scope is recorded explicitly.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts import joint_preprocessing_real_meta_bank as bank
from scripts.joint_preprocessing_synthetic_pilot import hash_file, json_write

# Reviewed catalog IDs: real observation tables, excluding simulation families,
# anonymous test uploads and obvious synthetic Kaggle example generators.
# Eligibility and duplicate-content checks still run on actual downloaded data.
REVIEWED_IDS = {
    2, 14, 15, 16, 18, 22, 23, 25, 28, 29, 30, 31, 32, 35, 36, 37,
    38, 39, 44, 49, 51, 53, 54, 57, 59, 151, 179, 180, 181, 182,
    185, 186, 188, 310, 311, 337, 357, 375, 451, 452, 454, 455, 458,
    466, 470, 473, 475, 488, 694, 717, 720, 728, 734, 735, 737, 750,
    757, 761, 770, 786, 798, 802, 810, 819, 821, 823, 825, 831,
    839, 841, 843, 844, 846, 847, 853, 858, 880, 886, 900, 906,
    907, 908, 909, 915, 930, 934, 940, 993, 1002, 1016, 1018, 1037,
    1044, 1046, 1048, 1049, 1050, 1053, 1056, 1063, 1065, 1067,
    1068, 1069, 1071, 1073, 1100, 1119, 1120, 1121, 1167, 1217,
    1242, 1443, 1444, 1446, 1447, 1451, 1452, 1453, 1461, 1466,
    1471, 1475, 1480, 1487, 1489, 1494, 1497, 1498, 1504, 1506,
    1508, 1510, 1511, 1523, 4154, 4340, 4538, 4541, 6332, 23381,
    23512, 23517, 40474, 40475, 40476, 40477, 40478, 40497, 40498,
    40589, 40663, 40672, 40685, 40691, 40700, 40701, 40705, 40707,
    40708, 40710, 40713, 40900, 40922, 40945, 40966, 40981, 40983,
    41146, 41150, 41156, 41160, 41162, 41168, 41430, 41440, 41701,
    41919, 41945, 41946, 42178, 42192, 42252, 42345, 42477, 42493,
    42544, 42585, 42750, 43044, 43097, 43255, 43439, 43551, 43595,
    43607, 43890, 43892, 43903, 43947, 43979, 43980, 44038, 44089,
    44149, 44162, 44224, 44226, 44227, 44232, 44234, 45023, 45035,
    45037, 45051, 45060, 45077, 45545, 45547, 45548, 45553, 45558,
    45560, 45562, 45577, 45578, 45711, 45712, 45714, 45717, 45748,
    45938, 46116, 46280, 46281, 46282, 46332, 46342, 46358, 46369,
    46372, 46382, 46416, 46420, 46421, 46422, 46431, 46435, 46441,
    46442, 46443, 46446, 46455, 46468, 46475, 46479, 46482, 46483,
    46519, 46525, 46526, 46528, 46531, 46532, 46535, 46536, 46537,
    46543, 46548, 46549, 46553, 46561, 46565, 46566, 46568, 46569,
    46570, 46590, 46593, 46595, 46597, 46600, 46601, 46603, 46605,
    46617, 46630, 46652, 46654, 46667, 46682, 46683, 46719, 46720,
    46733, 46760, 46764, 46810, 46820, 46825, 46827, 46828, 46846,
    46850, 46860, 46863, 46868, 46874, 46876, 46911, 46916, 46919,
    46920, 46924, 46930, 46935, 46936, 46937, 46938, 46940, 46950,
    46951, 46955, 46960, 46962, 47003, 47044, 47050, 47052, 47150,
    47151, 47152, 47153, 47171,
}

EXCLUDED_NAMES = {"higgs", "magictelescope", "magic", "estimationofobesitylevels",
                  "fitnessclub", "mobileprice", "ibmemployeeattrition", "ibmemployeeperformance"}

GROUP_ALIASES = {
    "uci_horse_colic": ["colic", "horse_colic_outcome"],
    "uci_contraceptive": ["cmc", "contraceptive_method"],
    "uci_heart_disease": ["heart-c", "heart-h", "heart-statlog", "heart_disease_cleveland", "cleveland", "cleve", "hungarian", "heart-statlog-uci", "Heart_disease_prediction_20"],
    "uci_thyroid": ["sick", "hypothyroid", "Sick_numeric", "allbp", "allrep", "dis", "thyroid-ann", "thyroid-allbp", "thyroid-allrep", "thyroid-allhyper", "thyroid-allhypo", "thyroid-dis"],
    "uci_multiple_features_digits": ["mfeat-fourier", "mfeat-karhunen", "mfeat-morphological", "mfeat-zernike", "mfeat-factors"],
    "uci_boston_housing": ["boston", "boston_corrected"],
    "uci_house_prices": ["houses", "house_16H", "house_8L"],
    "uci_cpu_performance": ["cpu_small", "cpu_act", "cpu_activity"],
    "uci_pbc": ["pbc", "pbcseq", "Cirrhosis_Patient_Survival_Prediction"],
    "ipums_census": ["kdd_ipums_la_97-small", "ipums_la_98-small", "ipums_la_99-small", "chscase_census2", "chscase_census3", "chscase_census4", "chscase_census5", "chscase_census6"],
    "nasa_software_defects": ["pc4", "pc3", "jm1", "mc1", "kc2", "kc3", "kc1", "pc1", "pc2", "mw1", "pc1_req"],
    "jedit_software_defects": ["jEdit_4.2_4.3", "jEdit_4.0_4.2"],
    "software_cost_estimation": ["PizzaCutter1", "PizzaCutter3", "CostaMadre1", "CastMetal1", "PieChart1", "PieChart2", "PieChart3", "Engine1"],
    "uci_california_housing": ["california", "California-Housing-Classification"],
    "uci_diabetes130": ["Diabetes130US", "Diabetes-130-Hospitals_(Fairlearn)"],
    "telco_customer_churn": ["telco-customer-churn", "blastchar"],
    "compas": ["compas-two-years", "compass"],
    "cpmp_2015": ["CPMP-2015-classification", "CPMP-2015-runtime-classification"],
    "uci_student_performance": ["students_scores", "1StudentPerfromance", "StudentsPerformance"],
    "uci_ilpd": ["ilpd", "ilpd-numeric"],
    "uci_heloc": ["heloc", "FICO-HELOC-cleaned"],
    "uci_hmeq": ["HMEQ_Data", "HMEQ_Data_New"],
    "apple_stock": ["Apple_Stock_Price_Trends_(2014-2023)", "Apple_Stock_Price_Trends_Classification", "Apple_Stock_Price_Trends"],
    "corporate_credit_ratings": ["Multiclass_Classification_for_Corporate_Credit_Ratings", "Corporate_Credit_Rating_Classification", "Corporate_Credit", "Corporate_Credit_Rating", "Corporate_Credit_Ratings"],
    "credit_score_classification": ["credit-score-classification-Hzl", "Credit_Score_Classification", "Credit_Score_Classification_downsampled", "dataset_credit_score"],
    "mortgage_ny": ["EDA-Home-Mortgage-NY", "EDA-Home-Mortgage-NY-2", "EDA-Home-Mortgage-NY-Sampled", "EDA-Home-Mortgage-NY-Sampled-Dataset"],
    "credit_card_fraud": ["Credit_Card_Fraud_Classification", "Credit_Card_Fraud", "Is_fraud", "Fraud-Detection-Updated)"],
    "loan_approval": ["Loan-Predication", "Loan_Approval_Status_Classification", "Loan_Status", "Loan_Approval_Status"],
    "osmi_mental_health": ["mental-health-in-tech-survey", "OSMI_Mental_Health_in_Tech_Survey"],
    "aztrees": ["aztrees3", "aztrees4"],
    "accidents_prediction": ["Accidents_Prediction_Dataset", "Accidents_Prediction_Dataset_Pipeline", "Accidents_Prediction_Balanced_v3"],
    "bwin": ["doa_bwin", "doa_bwin_balanced", "bwin_amlb"],
    "uci_pima_diabetes": ["diabetes", "DiabeticMellitus"],
    "uci_german_credit": ["credit-g", "German-Credit-Risk-with-Target", "Creditability-German-Credit-Data", "German-Credit-Data-Creditability", "German-Credit-Data-Creditability-2", "dataset_credit-g"],
    "uci_credit_approval": ["credit-approval", "Australian", "credit_approval_australia", "dataset_credit-approval"],
    "uci_vehicle_silhouettes": ["vehicle", "vehicleNorm"],
    "uci_bank_marketing": ["bank-marketing", "Bank_marketing_data_set_UCI"],
    "uci_wisconsin_breast_cancer": ["breast-w", "wdbc", "breast_cancer"],
    "uci_wine_quality": ["wine-quality-white", "wine-quality-red", "wine_quality_red", "wine_quality_white"],
}


def alias_map(old):
    result = {}
    for entries in old["candidates"].values():
        for e in entries:
            for name in [e["name"], *e.get("aliases", [])]:
                result[bank.normalized_name(name)] = e["source_group"]
    for group, names in GROUP_ALIASES.items():
        for name in names:
            result[bank.normalized_name(name)] = group
    return result


def build_catalog(snapshot, old):
    aliases = alias_map(old)
    eligible = []
    for row in snapshot["data"]["dataset"]:
        if int(row["did"]) not in REVIEWED_IDS or bank.normalized_name(row["name"]) in EXCLUDED_NAMES:
            continue
        q = {x["name"]: float(x["value"]) for x in row.get("quality", [])}
        if not (2 <= q.get("NumberOfClasses", 0) <= 10
                and 5 <= q.get("NumberOfFeatures", 0) - 1 <= 100
                and 256 <= q.get("NumberOfInstances", 0) <= 200000
                and q.get("NumberOfNumericFeatures", 0) >= 1):
            continue
        name = row["name"]
        group = aliases.get(bank.normalized_name(name), f"openml_{bank.normalized_name(name)}")
        eligible.append(dict(source="openml", data_id=int(row["did"]), name=name,
            source_group=group, catalog_quality=q,
            provenance="reviewed source name; OpenML classification version on real observations; detailed collection provenance not fully verified"))
    # Add known candidates absent from the reviewed catalog, including PMLB and
    # sklearn sources that were already audited in the preceding bank.
    identities = {(e["source"], e.get("data_id", e["name"])) for e in eligible}
    for entries in old["candidates"].values():
        for e in entries:
            if bank.normalized_name(e["name"]) in EXCLUDED_NAMES:
                continue
            key = e["source"], e.get("data_id", e["name"])
            if key not in identities:
                value = dict(e)
                value["source_group"] = aliases.get(bank.normalized_name(e["name"]), e["source_group"])
                value["provenance"] = "previously declared real-bank source; availability still checked"
                eligible.append(value)
                identities.add(key)
    # Hash order prevents catalog upload order/domain blocks determining the pool.
    import hashlib
    eligible.sort(key=lambda e: hashlib.sha256(("20261007:" + bank._entry_identity(e)).encode()).hexdigest())
    return dict(format_version=1, target_counts=dict(large=160, small=40, validation=25),
        seed=20261007, split_seeds=[0, 1], max_source_rows=16384,
        minimum_rows=256, max_episode_rows=1024,
        candidates=eligible,
        selection="availability/content audit, then metadata-balanced panel allocation; no model scores",
        scope="Real observation tables with existing classification targets; some OpenML targets are discretized. Known simulated/generated feature families excluded. Source names and declared aliases are not complete provenance proof.",
        reserved_test_policy="No test bank is generated/opened in this development experiment. Fresh confirmation sources will be chosen after freezing successful models.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--openml-catalog", type=Path, required=True)
    p.add_argument("--prior-candidates", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = build_catalog(json.loads(args.openml_catalog.read_text()), json.loads(args.prior_candidates.read_text()))
    result["catalog_sha256"] = hash_file(args.openml_catalog)
    result["prior_candidates_sha256"] = hash_file(args.prior_candidates)
    json_write(args.output, result)
    print(f"Frozen {len(result['candidates'])} candidates, {len({bank.normalized_name(e['source_group']) for e in result['candidates']})} declared source groups: {args.output}")


if __name__ == "__main__":
    main()
