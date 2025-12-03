# Dataset folder
# Place your RFMiD dataset here with the following structure:
#
# dataset/
# ├── Training_Set/
# │   ├── Training/          (or images/ - folder containing training images)
# │   └── RFMiD_Training_Labels.csv
# ├── Validation_Set/
# │   ├── Validation/        (or images/ - folder containing validation images)
# │   └── RFMiD_Validation_Labels.csv
# └── Test_Set/
#     ├── Test/              (or images/ - folder containing test images)
#     └── RFMiD_Testing_Labels.csv
#
# The CSV files should have columns:
# - ID: image filename (without extension)
# - Disease_Risk: 0 or 1
# - DR, ARMD, MH, DN, ... (disease columns): 0 or 1 each
