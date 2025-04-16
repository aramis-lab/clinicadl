from pathlib import Path

from clinicadl.IO.bids import Bids

bids = Bids(path=Path("/Users/camille.brianceau/aramis/DATA/BIDS_QC"))
bids.load()
print(bids.subjects_list)
print(bids.participants_tsv)
print(bids.subjects)
print(bids.subjects["ADNI011S0002"].sessions_list)
print(bids.subjects["ADNI011S0002"].sessions)
print(bids.subjects["ADNI011S0002"].id)
print(bids.subjects["ADNI011S0002"].bids_dir)
print(
    bids.subjects["ADNI011S0002"].sessions["M00"].scans_tsv
)  # Example of accessing the scans.tsv file path
print(
    bids.subjects["ADNI011S0002"].sessions["M00"].filename_list
)  # Example of accessing the files list
print(bids.subjects["ADNI011S0002"].sessions["M00"].bids_dir)
print(bids.subjects["ADNI011S0002"].sessions["M00"].subject)
print(bids.subjects["ADNI011S0002"].sessions["M00"].subject_dir)
print(bids.subjects["ADNI011S0002"].sessions["M00"].id)
print(bids.subjects["ADNI011S0002"].sessions["M00"].file_types)

print("end")
